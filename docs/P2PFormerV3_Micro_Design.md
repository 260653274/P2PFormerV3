# Architecture: P2PFormerV3 — Contour-Conditioned Support-Box Diffusion

> 状态：Micro Design v0.1（2026-08-02）
> 前置条件：`P2PFormerV3 Macro Design` 已由用户明确批准。
> 设计目标：在不同时重写 detector 与 contour conditioner 的前提下，以最小可证伪架构验证 primitive-level support-box diffusion 是否成立。
> 一手依据：用户提供的 [P2PFormerV2 PDF](../../.memory/artifacts/P2PFormerV2_Improving_Primitive-Based_Regular_Building_Contour_Extraction_Methods_via_Contour_Feature_Enhancement.pdf)，重点为 Fig. 2–8、Eq. (1)–(7) 与 Table V–VII。

## 0. 设计结论

首个可实现版本固定为：

> **V2-compatible contour conditioner + 40-slot 4D AABB diffusion + final 6D corner recovery + successor-cycle topology reconciliation。**

本版本刻意不加入 rotated-box diffusion、categorical validity diffusion、topology diffusion、scene-level box diffusion、CFG、自条件或 consistency distillation。原因不是这些机制无效，而是它们会同时改变过多变量，使“扩散是否优于一步几何回归”无法归因。

核心数据流如下：

~~~mermaid
flowchart LR
    I["RGB [B,3,H,W]"] --> S["共享场景特征"]
    S --> D["FCOS 建筑框"]
    S --> C["V2-compatible 轮廓条件器"]
    D --> C
    C -->|"P4/P3/P2 条件记忆<br/>3×[N,64,256]"| G["4D support-set 去噪器"]
    Z["Gaussian state<br/>[N,40,4]"] --> G
    G -->|"clean support boxes<br/>[N,40,4]"| P["6D corner + validity"]
    C -->|"masked dense map<br/>[N,256,32,32]"| P
    P -->|"valid corner set"| T["successor topology"]
    T --> O["closed polygon + mask + score"]
~~~

这里的 `N` 是一个 batch 展平后的建筑实例总数，`N=Σ_b N_b`。图像特征与 V2 条件只计算一次；2/4 个扩散步只重复轻量 set denoiser。

---

## 1. Requirements & Objectives

### 1.1 Requirements

- **应用场景**：高分辨率遥感影像中的规则建筑实例矢量轮廓提取。
- **输入**：RGB 影像 `I∈R^{B×3×H×W}`；代表性 WHU-Mix 设置为 `H=W=640`。
- **训练标注**：building box、polygon/mask；polygon 转换为 `[previous,current,next]` corner primitives。
- **输出**：
  - 建筑实例框与实例置信度；
  - 每实例可变长 polygon `P_i∈R^{K_i×2}`；
  - COCO-compatible raster mask；
  - primitive validity 与 topology quality。

### 1.2 Objectives

- **主基线**：忠实复现的 P2PFormerV2，而不是仅与 V1 比较。
- **质量目标**：优先提高 AP75、Boundary IoU/PoLiS、corner recall，并降低自交、重复点和退化 polygon。
- **鲁棒性目标**：在 building box 中心、尺度、比例扰动及弱边界情况下保持 primitive recall。
- **效率目标**：2 NFE 为默认设置，4 NFE 为 quality setting；不采用需要两次前向的 CFG。
- **参数目标**：总参数不高于当前 R50 P2PFormer V1 的实测 `41.28 M`。
- **科学门槛**：若同容量一步 refiner 与 2/4-step diffusion 持平，则删除 diffusion 叙事，保留 support-box 与 topology 设计。

---

## 2. Approved Macro Design

### 2.1 Functional graph

~~~mermaid
flowchart LR
    A["场景证据提取"] --> B["建筑实例定位"]
    A --> C["轮廓证据整合"]
    B --> C
    B --> D["几何支撑集合初始化"]
    C --> E["条件化几何支撑重建"]
    D --> E
    E --> F["基元恢复与有效性估计"]
    C --> F
    F --> G["拓扑协调"]
    G --> H["多边形装配与输出投影"]
~~~

### 2.2 Sub-block interfaces

| # | Sub-block | Input | Output |
|---|---|---|---|
| 1 | Shared scene evidence | RGB image | `P2,P3,P4,P5` multiscale maps |
| 2 | Building localization | `P3,P4,P5` | building RoIs `R:[N,5]` |
| 3 | Contour evidence conditioning | `P2,P3,P4,R` | three contour memories, deformed positions, dense Feature Mask |
| 4 | Support-set construction/reconstruction | anchors/noise, contour memories, timestep | clean 4D support boxes |
| 5 | Primitive recovery | clean supports, dense masked contour map | 6D corner primitives and validity |
| 6 | Topology reconciliation | active primitives and pairwise successor scores | one clockwise, non-self-intersecting cycle |
| 7 | Output projection | cycle, RoI, image metadata | vector polygon, mask and score |

---

## 3. Micro Design

### 3.0 Global notation and fixed constants

| Symbol | Default | Meaning |
|---|---:|---|
| `B` | variable | image batch size |
| `N` | variable | total building RoIs across the batch |
| `M` | 40 | support slots per building |
| `D` | 256 | hidden/channel dimension |
| `h` | 8 | attention heads |
| `d_h` | 32 | dimension per head |
| `S` | 32 | fixed instance crop side |
| `L_c` | 64 | contour condition tokens per native scale |
| `T` | 1000 | training diffusion timesteps |
| NFE | 2 / 4 | default / quality inference denoising calls |

`M=40` is data-driven. On the local WHU-Mix training JSON, after removing the repeated closure point, the raw component vertex-count quantiles are `p99=30` and `p99.5=38`; 40 slots cover `99.576%` of components. The `M=30` setting must still be reported for direct V1/V2 comparison. Samples above 40 are never silently truncated: they use topology-preserving simplification to 40 vertices and are reported as a separate complexity subset.

For `K>40`, use closed-curve Visvalingam–Whyatt simplification: repeatedly remove the vertex with the smallest effective triangle area, but reject any removal whose replacement edge intersects a non-adjacent edge or flips polygon winding. Stop at 40 vertices. Report both the full test set and the untouched `K≤40` subset so this rare preprocessing case cannot hide a capacity failure.

### 3.1 Shared Scene Feature Extraction and Building Localization

**Implementation family:** retain the current ResNet-50 + multiscale deformable neck + FCOS path. This branch is already trained for the target domain and is not the research variable of V3.

**Internal connectivity:**

~~~mermaid
flowchart LR
    I["[B,3,640,640]"] --> R["ResNet-50 + DCNv2"]
    R --> N["multiscale deformable neck"]
    N --> P2["P2 [B,256,160,160]"]
    N --> P3["P3 [B,256,80,80]"]
    N --> P4["P4 [B,256,40,40]"]
    N --> P5["P5 [B,256,20,20]"]
    P3 --> F["FCOS"]
    P4 --> F
    P5 --> F
    F --> ROI["RoIs [N,5]"]
~~~

**Tensor flow:**

| Step | Input | Operation | Output | Information effect |
|---|---|---|---|---|
| 1 | `[B,3,640,640]` | ResNet stem + four stages | C2–C5 at strides 4/8/16/32 | hierarchical local evidence |
| 2 | C2–C5 | 3-layer multiscale deformable aggregation | `P2…P5`, all `C=256` | aligns semantics across scales |
| 3 | `P3,P4,P5` | FCOS classification/regression/centerness | per-image building boxes | compresses dense evidence to instance proposals |
| 4 | per-image boxes | concatenate batch indices | `R:[N,5]` | preserves instance/image ownership |

At inference, `R` comes from FCOS after NMS. During training it follows the exposure-gap curriculum in Section 3.8; the `FCOS→conditioner` edge therefore means the inference path plus the detached 10% matched-proposal branch, not end-to-end gradient through proposal selection.

**Decision defense:**

- **Why retain FCOS:** simultaneously diffusing scene boxes and primitive boxes would confound two proposal levels and make the V3 contribution unauditable.
- **Why keep P2–P4 for contours:** the V2 PDF Fig. 3 explicitly routes P2/P3/P4 to the contour enhancer and P3/P4/P5 to detection.
- **Inductive bias:** convolutional locality and multiscale translation equivariance are appropriate for limited-data remote-sensing localization.

**Normalization / activation / initialization:** inherited unchanged from the current checkpoint.

**Parameters:** exact local build: backbone `24.379 M`, neck `3.839 M`, FCOS head `4.863 M`; shared subtotal `33.082 M`.

### 3.2 V2-Compatible Contour Evidence Conditioner

#### 3.2.1 What the PDF fixes and what this design completes

The V2 PDF establishes:

- Fig. 2(a): V1 uniform RoI-Align contains many ineffective samples.
- Fig. 2(b): V2 deforms `8×8` reference points toward contours.
- Fig. 2(c): Feature Mask suppresses invalid features.
- Eq. (4): the mask is a narrow contour buffer, not an interior building mask; masked dense features are re-sampled at the **same deformed positions** and concatenated with sparse features.
- Default sparse scale `s=4`; contour buffer extends 8 original-image pixels inward and outward.

The source code is unavailable, so the following completion choices are explicit V3 engineering decisions:

| Missing PDF detail | V3 completion |
|---|---|
| variable crop batching | RoIAlign every native level to `32×32` before attention |
| offset stride | shared `DWConv5×5,stride=4,pad=2` |
| `s=4` coordinate unit | four cells in the `32×32` RoI grid |
| DFE “MHSA” with different q and k/v | one pre-norm multihead **cross-attention** block |
| P2/P3/P4 fusion | learned softmax-weighted sum |
| sigmoid vs two-class CE conflict | two logits + softmax CE, following Eq. (6) |
| hard threshold/dilation | differentiable soft contour-band dilation |
| hard zeroing | residual gate floor `λ_mask=0.1` |
| `512→256` after concat | shared linear projection |
| deformed position to decoder | 2-D sine encoding added to keys, never values |

#### 3.2.2 Internal connectivity

~~~mermaid
flowchart TD
    IN["P2/P3/P4 + RoIs"] --> ROI["per-level RoIAlign<br/>X_l [N,256,32,32]"]
    ROI --> OFF["shared DWConv5×5/s4 → GELU → Conv1×1 → 4×tanh<br/>Δp_l [N,2,8,8]"]
    OFF --> POS["permute; p0 + Δp_l<br/>Z_l [N,8,8,2]"]
    ROI --> SAMPLE["bilinear sample at Z_l"]
    POS --> SAMPLE
    SAMPLE --> SP["sparse tokens S_l [N,64,256]"]

    ROI --> Q["dense Q_l [N,1024,256]"]
    SP --> KV["sparse K_l,V_l [N,64,256]"]
    Q --> CA["8-head cross-attention"]
    KV --> CA
    CA --> YL["Y_l [N,256,32,32]"]

    YL --> MS["softmax-weighted P2/P3/P4 sum"]
    MS --> MH["Conv3×3 256→64 → GELU → Conv1×1 64→2"]
    MH --> CM["contour probability [N,1,32,32]"]
    CM --> BAND["scale-aware soft dilation<br/>B [N,1,32,32]"]

    ROI --> MASK["X_l × (0.1 + 0.9B)"]
    BAND --> MASK
    MASK --> RS["re-sample at Z_l<br/>D_l [N,64,256]"]
    POS --> RS
    SP --> CAT["Concat [N,64,512]"]
    RS --> CAT
    CAT --> PROJ["Linear 512→256"]
    PROJ --> MEM["E_l [N,64,256]"]
    POS --> PE["2-D sine PE [N,64,256]"]
    MS --> DENSE["residual-mask gate<br/>C32 [N,256,32,32]"]
    BAND --> DENSE
    MS --> GI["GAP<br/>c_img [N,256]"]
~~~

#### 3.2.3 Tensor flow

For each native level `l∈{2,3,4}`:

| Step | Input Shape | Operation | Output Shape | Why |
|---|---|---|---|---|
| 1 | `P_l:[B,256,H_l,W_l]`, `R:[N,5]` | aligned RoIAlign `32×32`, sampling ratio 2 | `X_l:[N,256,32,32]` | fixed batching and cross-scale alignment |
| 2 | `X_l` | DWConv `5×5,s=4` | `[N,256,8,8]` | local offset evidence with low parameters |
| 3 | `[N,256,8,8]` | GELU → Conv `1×1,256→2` → `4·tanh` | `Δp_l:[N,2,8,8]` | offsets bounded to four `32×32` crop pixels |
| 4 | `p_0,Δp_l` | add in crop pixels, clip, convert to normalized grid | `Z_l:[N,8,8,2]` | deformed contour sample coordinates |
| 5 | `X_l,Z_l` | bilinear grid sample | `[N,256,8,8]` | differentiable sparse sampling |
| 6 | previous | flatten + transpose | `S_l:[N,64,256]` | information-preserving spatial-to-token reshape |
| 7 | `X_l` | flatten + transpose | `Q_l:[N,1024,256]` | dense spatial queries |
| 8 | `Q_l` | Linear Q, split heads | `[N,8,1024,32]` | attention queries |
| 9 | `S_l` | Linear K/V, split heads | `[N,8,64,32]` each | sparse contour evidence |
| 10 | Q,K | scaled dot product | `[N,8,1024,64]` | every dense location queries 64 contour samples |
| 11 | scores,V | weighted sum + merge + projection | `[N,1024,256]` | sparse-to-dense interaction |
| 12 | `X_l` + attention output | residual add, reshape | `Y_l:[N,256,32,32]` | retains original crop if sparse evidence is imperfect |

Across levels:

| Step | Input Shape | Operation | Output Shape |
|---|---|---|---|
| 13 | three `Y_l` | scalar softmax-weighted sum | `Y:[N,256,32,32]` |
| 14 | `Y` | Conv `3×3,256→64` + GELU | `[N,64,32,32]` |
| 15 | previous | Conv `1×1,64→2` + softmax | `P_c:[N,2,32,32]` |
| 16 | contour channel | scale-aware soft dilation | `B_c:[N,1,32,32]` |
| 17 | `X_l,B_c` | `X_l⊙(0.1+0.9B_c)` | `X_l^m:[N,256,32,32]` |
| 18 | `X_l^m,Z_l` | bilinear sample | `D_l:[N,64,256]` |
| 19 | `S_l,D_l` | channel concat | `[N,64,512]` |
| 20 | previous | shared Linear `512→256` | `E_l:[N,64,256]` |
| 21 | `Z_l` | 2-D sine encoding | `PE_l:[N,64,256]` |

For `grid_sample(align_corners=False)`, the regular sample center at lattice index `(u,v)` is

`p_0(u,v)=(4u+1.5,4v+1.5),  u,v∈{0,…,7}`.

The offset subnet predicts channels `(o_x,o_y)`; `u` indexes width/x, `v` indexes height/y, and `Z_l[...,0:2]` follows grid-sample's `(x,y)` order. Sampling uses crop-pixel coordinate `p=p_0+4·tanh(o_l)`, clipped to `[0,31]`, then converts it to the normalized grid as `Z_l=2(p+0.5)/32-1`. This resolves the paper's unspecified offset coordinate convention while retaining its reported scale `s=4`.

The contour-band half-width is defined in original pixels, not feature cells:

`r_x=8·32/w_R,  r_y=8·32/h_R`,

where `w_R,h_R` are RoI width and height in original-image pixels. Clamp both radii to `[1,16]` cells. For reproducibility, form an elliptical neighborhood `Ω_R`, unfold the contour channel over the maximum `33×33` window, mask samples outside `Ω_R`, and use the temperature-`0.1` soft maximum

`B_c(i)=Σ_{δ∈Ω_R} softmax(P_c(i+δ)/0.1)_δ P_c(i+δ)`.

Zero padding is excluded from the softmax. This remains in `[0,1]`, is differentiable with respect to the contour probabilities and approaches morphological dilation. The default V3 gate is

`X_l^m=X_l⊙[0.1+0.9B_c]`.

The `0.1` residual floor is load-bearing: an incorrect early contour prediction can suppress, but cannot completely delete, weak boundary evidence. The mandatory hard V2-semantics baseline uses the explicit completion `B_hard=DilateEllipse(1[P_c>0.5],Ω_R)` and `X_l^m=X_l⊙B_hard`; the PDF fixes the hard contour buffer but does not publish its threshold.

The contour supervision is a boundary target, not a filled-building mask: transform the original GT boundary segments into the selected outer RoI, clip only the segment portions outside the crop, and rasterize a one-cell-wide 8-connected polyline on the `32×32` grid. Never close a cropped polygon along the RoI border, which would create a false contour. Every non-boundary crop cell is background; there is no ignore band, and CE is averaged over all `1024` cells and then over RoIs. Use the paper's ordinary two-class pixel CE first; class reweighting is allowed only as a reported ablation if the positive class collapses.

#### 3.2.4 Condition outputs

The conditioner returns:

- coarse-to-fine memories `(E_4,E_3,E_2)`, each `[N,64,256]`;
- their deformed positional encodings `(PE_4,PE_3,PE_2)`;
- masked dense map `C32=Y⊙(0.1+0.9B_c):[N,256,32,32]`;
- global instance context `c_img=GAP(Y):[N,256]`;
- contour logits `[N,2,32,32]` for auxiliary supervision.

The reverse-reproduction ablations keep the decoder interface fixed:

- **SFE-only:** learn `Z_l`, bypass DFE/mask, and use `E_l=S_l`.
- **DFE-only:** fix `Z_l=p_0`, use uniform samples as DFE K/V, predict the contour mask, and use regular-position masked samples `E_l=D_l`.
- **SFE+DFE:** use the full same-position concat and `512→256` projection.

All three routes return exactly `3×[N,64,256]` plus positional encodings, so Table V trends are not confounded by a different decoder or token count.

**Key module defenses:**

- **Sparse offset branch:** V2 Table V shows SFE-only gains are more consistent than DFE-only gains. It moves limited samples toward the signal rather than increasing query count.
- **Dense contour branch:** directly supervises where contour evidence should exist and makes Feature Mask auditable.
- **Same-position resampling:** strictly necessary to preserve spatial correspondence between sparse and masked-dense tokens; sampling at different positions makes concatenation semantically invalid.
- **Soft residual mask:** prevents false-negative contour maps from starving the diffusion branch; hard masking is retained only as an ablation.
- **No DFE FFN stack:** the block only transports sparse contour evidence into dense queries. Additional depth has no demonstrated role and inflates per-RoI cost.

**Normalization / activation / initialization:**

- Cross-attention: pre-LayerNorm; stable for variable `N` and small per-GPU batches.
- Activations: GELU, matching the V2 offset subnet.
- Offset Conv `1×1`: zero initialization, so initial sampling is the regular `8×8` grid.
- Multiscale fusion weights: initialized equally.
- Mask logits: standard Xavier initialization.

**Parameters / compute:** approximately `0.55 M` parameters and `0.70 GMAC` per RoI. The same CFE weights are shared across P2/P3/P4; scale identity is retained by the routed outputs `(E_4,E_3,E_2)` and the learned scalar fusion weights, so an extra level embedding is unnecessary.

### 3.3 Support Target Construction, Stable Coupling and Diffusion

#### 3.3.1 Corner support target

For a normalized V1 corner primitive under a well-covered outer RoI

`ℓ_j=(p^-_j,p_j,p^+_j)∈[0,1]^6`,

the same construction remains valid for slightly under-covered training RoIs, where coordinates may extend to `[-0.25,1.25]`. Samples beyond this guard range are rejected and redrawn rather than silently clipped.

the support box must cover only local evidence near `p_j`, not the full adjacent edges. Define clipped incident points:

`q_j^±=p_j+min(1,ρ/max(||p_j^±-p_j||_2,ε))(p_j^±-p_j),  ρ=6/32.`

Take the AABB of `{q_j^-,p_j,q_j^+}`, expand every side by `δ=2/32`, and enforce minimum width/height `4/32`. The result is

`b_j=(c_x,c_y,w,h)∈R^4.`

The diffusion state is

`g_j=[c_x,c_y,log(w+ε),log(h+ε)]`

and is standardized using training-set active-support statistics:

`z_j=(g_j-μ_g)/σ_g.`

**Why 4D AABB first:** the support only chooses a local evidence region. Corner orientation remains represented by the final three-point primitive. Adding angle would introduce periodicity, rotated sampling and a second direction ambiguity before 4D boxes have demonstrated value.

#### 3.3.2 Fixed clean coupling

Use an `8×5` anchor lattice `A={a_m}_{m=1}^{40}` over the normalized RoI, with base box size `8/32×8/32`. Anchors define spatial slot identity but do not constrain how far a predicted support may move.

For each clean instance, solve Hungarian matching once:

`C_{mj}=4||c(a_m)-c(b_j)||_1 + 2[1-GIoU(a_m,b_j)] + ||log s(a_m)-log s(b_j)||_1.`

- Active slot: clean target `z_{0,m}=z(b_j)`, validity `y_m=1`.
- Null slot: clean target is its anchor geometry `z(a_m)`, validity `y_m=0`.
- Primitive target and clockwise/counter-clockwise successor targets use the same coupling.
- The coupling is computed from clean GT only and is reused for every `t`; per-timestep Hungarian matching is forbidden.
- Anchor-state pairs are randomly permuted together during training, so the network remains equivariant to list order while retaining spatial identity.

Null geometry receives only `0.1×` the active geometry weight. This gives DDIM a complete clean target for every slot without allowing padding to dominate training.

#### 3.3.3 Forward and reverse process

Use cosine variance-preserving diffusion with `T=1000`. For `t∼Uniform{1,…,999}`:

`z_t=√(ᾱ_t)z_0+√(1-ᾱ_t)ε,   ε∼N(0,I),`

with shape `[N,40,4]`.

The denoiser directly predicts clean state:

`ẑ_0=f_θ(z_t,t,A,E_4,E_3,E_2,c_img).`

At inference, initialize `z_999∼N(0,I)` once per RoI and use a single sample. Benchmark runs must fix the RNG seed and report no best-of-`K` selection or ensemble.

Direct `x_0` prediction is chosen because box L1/GIoU and support containment are defined in clean geometry. It also follows the low-dimensional box-generation precedent of [DiffusionDet](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf).

Deterministic DDIM update from `t` to `s<t`:

`ε̂=(z_t-√ᾱ_t ẑ_0)/√(1-ᾱ_t),`

`z_s=√ᾱ_s ẑ_0+√(1-ᾱ_s)ε̂.`

Inference schedules:

- 1 NFE diagnostic: `999→clean`.
- 2 NFE default: `999→499→clean`.
- 4 NFE quality: `999→749→499→249→clean`.

No learned variance, CFG, stochastic ensemble or self-conditioning is used in v0.1.

**Parameters:** none; target construction and DDIM are parameter-free.

### 3.4 Coarse-to-Fine Set Denoiser

**Implementation family:** a small set Transformer with three coarse-to-fine blocks. Self-attention models interactions among candidate supports; cross-attention queries cached contour evidence. Full image DiT/U-Net is unnecessary because the diffusion state is only `40×4`.

#### 3.4.1 Input embedding

| Input | Operation | Output |
|---|---|---|
| `z_t:[N,40,4]` | MLP `4→128→256`, SiLU | `E_z:[N,40,256]` |
| `A:[1,40,4]` | MLP `4→128→256`, SiLU | `E_a:[1,40,256]` |
| `E_z,E_a` | broadcast add | `H_0:[N,40,256]` |
| scalar log-SNR `t` | 128-D sinusoidal embedding + MLP `128→256→256`, SiLU | `e_t:[N,256]` |
| `c_img:[N,256]` | Linear `256→256` | `e_i:[N,256]` |
| `e_t,e_i` | add | block condition `c:[N,256]` |

There is no arbitrary learned slot-ID embedding. Spatial anchors provide the only identity, and permuting `(z_t,A)` together permutes the output.

The state and anchor MLPs use separate weights. Together with the specified time MLP and image-context Linear, these input embeddings contain approximately `0.23 M` parameters.

#### 3.4.2 One denoising block

The three blocks use different weights and condition on `E_4→E_3→E_2`. The whole three-block denoiser is reused at every DDIM step.

~~~mermaid
flowchart TD
    H["H_l [N,40,256]"] --> MOD1["AdaLN(H;c)"]
    C["c [N,256]"] --> AZ["zero-init Linear → 9D modulation"]
    AZ --> MOD1
    MOD1 --> SA["slot self-attention"]
    SA --> G1["gate_sa × output"]
    H --> A1((+))
    G1 --> A1

    A1 --> MOD2["AdaLN(;c)"]
    AZ --> MOD2
    MOD2 --> Q["Q [N,8,40,32]"]
    MEMK["E_l + PE_l<br/>[N,64,256]"] --> K["cached K [N,8,64,32]"]
    MEMV["E_l<br/>[N,64,256]"] --> V["cached V [N,8,64,32]"]
    Q --> CA["cross-attention"]
    K --> CA
    V --> CA
    CA --> G2["gate_ca × output"]
    A1 --> A2((+))
    G2 --> A2

    A2 --> MOD3["AdaLN(;c)"]
    AZ --> MOD3
    MOD3 --> FFN["Linear 256→1024<br/>GELU<br/>Linear 1024→256"]
    FFN --> G3["gate_ffn × output"]
    A2 --> OUT((+))
    G3 --> OUT
~~~

AdaLN-Zero for each residual branch is:

`AdaLN(x;c)=[1+γ(c)]LN(x)+β(c),`

and the branch output is multiplied by a zero-initialized gate `a(c)`. Three branches require `3×(γ,β,a)=9D` modulation values per block.

#### 3.4.3 Tensor flow inside one block

| Step | Input Shape | Operation | Output Shape | Why |
|---|---|---|---|---|
| 1 | `H_l:[N,40,256]` | AdaLN | `[N,40,256]` | time/image-conditioned normalization |
| 2 | previous | Linear Q/K/V | three `[N,40,256]` | slot relation streams |
| 3 | each stream | split 8 heads | `[N,8,40,32]` | subspace specialization |
| 4 | Q,K | scaled dot product | `[N,8,40,40]` | pairwise support interaction |
| 5 | scores,V | weighted sum, merge, output projection | `[N,40,256]` | set-coherent support features |
| 6 | prior `H_l` | gated residual add | `[N,40,256]` | identity path at initialization |
| 7 | updated slots | AdaLN + Q projection | `[N,8,40,32]` | image-conditioned queries |
| 8 | `E_l+PE_l` | K projection, split | `[N,8,64,32]` | positions affect addressing |
| 9 | `E_l` | V projection, split | `[N,8,64,32]` | values carry contour evidence without position contamination |
| 10 | Q,K | scaled dot product | `[N,8,40,64]` | each support queries one native contour scale |
| 11 | scores,V | weighted sum, merge, projection | `[N,40,256]` | conditioned geometry evidence |
| 12 | previous | gated residual add | `[N,40,256]` | preserves support state |
| 13 | updated slots | AdaLN + Linear `256→1024` | `[N,40,1024]` | channel expansion |
| 14 | previous | GELU + Linear `1024→256` | `[N,40,256]` | nonlinear per-slot refinement |
| 15 | previous | gated residual add | `H_{l+1}:[N,40,256]` | stable gradient highway |

After block 3:

`ẑ_0=Linear(LN(H_3))∈R^{N×40×4}.`

The head is unbounded in standardized space. Decoding to box space applies dataset mean/std, exponentiates width/height and clips only for sampling/evaluation.

Explicitly, `ĝ=σ_g⊙ẑ_0+μ_g`, `w=max(exp(ĝ_3)-ε,ε)`, `h=max(exp(ĝ_4)-ε,ε)`, `b̂=(ĝ_1,ĝ_2,w,h)`, and `xyxy=(c_x-w/2,c_y-h/2,c_x+w/2,c_y+h/2)`. GIoU uses this decoded `xyxy`; DDIM always updates the standardized `z` state.

**Key module defenses:**

- **Slot self-attention:** duplicate and missing supports are set-level phenomena; independent per-slot MLPs cannot coordinate them.
- **Coarse-to-fine cross-attention:** P4 supplies stable context first; P2 supplies precise boundary evidence last. Three blocks align exactly with three native contour levels.
- **Condition K/V cache:** image evidence is constant across DDIM steps, so recomputing projections is wasteful.
- **AdaLN-Zero:** the same denoiser is called recursively; identity-initialized residuals prevent early training from amplifying state errors across steps.
- **No per-step RoIAlign on noisy supports:** high-noise boxes frequently cover background. Current geometry only forms Q; stable V2 evidence remains K/V.

**Normalization / activation / initialization:**

- Pre-LayerNorm in all attention and FFN branches.
- GELU in FFN; SiLU in time/state embedding MLPs.
- QKV/output: Xavier uniform.
- All AdaLN residual gates and final modulation projection: zero initialization.
- Final `ẑ_0` head: small normal initialization (`σ=0.01`).

**Parameters / compute:** approximately `5.15 M` parameters. One three-block denoiser call is approximately `0.118 GMAC/RoI` after condition K/V caching; cache construction is approximately `0.025 GMAC/RoI` once.

### 3.5 Final Primitive Recovery and Validity

The support boxes select local evidence only after the final denoising step.

~~~mermaid
flowchart LR
    B["predicted clean boxes [N,40,4]"] --> GRID["axis-aligned 4×4 grids"]
    GRID --> GS["differentiable grid_sample"]
    C["C32 [N,256,32,32]"] --> GS
    GS --> R["[N,40,256,4,4]"]
    R --> GAP["GAP [N,40,256]"]
    H["H3 [N,40,256]"] --> ADD((+))
    GAP --> WR["Linear 256→256"]
    WR --> ADD
    ADD --> LN["LayerNorm"]
    LN --> PH["MLP 256→256→6<br/>1.5×sigmoid−0.25"]
    LN --> VH["Linear 256→1"]
    PH --> P["corner triples [N,40,6]"]
    VH --> V["validity logits [N,40]"]
~~~

**Tensor flow:**

| Step | Input | Operation | Output |
|---|---|---|---|
| 1 | `C32:[N,256,32,32]`, `b̂:[N,40,4]` | axis-aligned grid construction + `grid_sample` `4×4` | `[N,40,256,4,4]` |
| 2 | previous | spatial GAP | `r:[N,40,256]` |
| 3 | `H_3,r` | `LN(H_3+W_r r)` | `F:[N,40,256]` |
| 4 | `F` | MLP `256→256→6` + `1.5·sigmoid−0.25` | `ℓ̂:[N,40,6]` |
| 5 | `F` | Linear `256→1` | validity logits `v:[N,40]` |

`ℓ̂=[p^-,p,p^+]` remains in normalized instance coordinates, has representable range `[-0.25,1.25]`, and retains V1's reversal symmetry. The extended range handles mild detector under-coverage without forcing impossible `[0,1]` targets. The support box is not forced to contain full adjacent vertices; it only needs to contain the center corner and local incident-edge evidence.

For support `b̂_m=(c_x,c_y,w,h)`, grid point `(u,v)` is

`G_{muv}=(c_x+(u-1.5)w/4, c_y+(v-1.5)h/4),  u,v∈{0,1,2,3}`.

Convert `G` to the `[-1,1]` convention and call `grid_sample(align_corners=False)`. Unlike the common RoIAlign kernels used by this repository, this explicit grid is differentiable with respect to box coordinates; therefore primitive loss can refine support geometry. Coordinates are clipped only at the sampler boundary, and the out-of-bound rate must be logged.

**Why final-only local sampling:** it makes support geometry operational while avoiding the high-noise sampling failure that motivated V2.

**Why no primitive type head:** the main V2 experiments use corners, and adding point/line/type classification creates a second discrete problem before corner support diffusion is validated.

**Parameters / compute:** approximately `0.14 M` parameters and `<0.01 GMAC/RoI`.

### 3.6 Successor Topology Reconciliation

Independent 36-bin order classification is replaced by a pairwise successor matrix.

Let the predicted corner center be `p_m=ℓ̂_m[2:4]`, outgoing direction `d_m^+=p_m^+-p_m` and incoming direction `d_m^-=p_m-p_m^-`.

~~~mermaid
flowchart LR
    F["slot features F [N,40,256]"] --> Q["Linear 256→64"]
    F --> K["Linear 256→64"]
    Q --> DOT["QKᵀ/√64 [N,40,40]"]
    P["corner geometry"] --> GEO["pairwise geometry MLP"]
    GEO --> BIAS["bias [N,40,40]"]
    DOT --> ADD((+))
    BIAS --> ADD
    ADD --> SK["5-step log-Sinkhorn"]
    SK --> TRAIN["successor NLL"]
    ADD --> HUN["inference Hungarian cycle cover"]
    HUN --> MERGE["subtour 2-edge merge"]
    MERGE --> OPT["2-opt crossing removal"]
    OPT --> POLY["single clockwise polygon"]
~~~

Pairwise logits:

`E_{mn}=(W_qF_m)^T(W_kF_n)/√64 + MLP([p_n-p_m,d_m^+,d_n^-,||p_n-p_m||_2]).`

Shapes:

| Step | Input | Operation | Output |
|---|---|---|---|
| 1 | `F:[N,40,256]` | two Linear `256→64` | Q,K `[N,40,64]` |
| 2 | Q,K | batched matmul | `[N,40,40]` |
| 3 | pairwise 7-D geometry | MLP `7→64→1` | `[N,40,40]` |
| 4 | content + geometry | add, diagonal `−∞` | successor logits `E` |
| 5 | active `K×K` submatrix | five row/column `logsumexp` normalizations | log-probability matrix `log P:[N,K,K]` |

Training uses GT active slots. If `s_cw(m)` and `s_ccw(m)` are the two target successors, the loss is the lower mean edge NLL:

`L_topo=min[-K^{-1}Σ_m log P_{m,s_cw(m)}, -K^{-1}Σ_m log P_{m,s_ccw(m)}].`

Inference:

1. Keep `sigmoid(v)>0.5`; fewer than three means invalid instance.
2. Suppress duplicate centers within `1/64` normalized distance by validity score.
3. Hungarian maximum assignment gives one incoming and one outgoing edge per active node.
4. If the assignment produces multiple subtours, merge them by the minimum-cost 2-edge swap.
5. Apply 2-opt until no segment intersection remains or a fixed iteration cap is reached.
6. Normalize winding to clockwise and emit corner centers as the polygon; reject and count the instance if intersections remain at the iteration cap.

The hard solver is deliberately outside backpropagation. Its neural predecessor is trained by `L_topo`; the implementation must report subtour count, merge count, 2-opt count and failure rate.

**Why this is necessary:** a rank/bin predicts where a primitive lies in an ordering but does not enforce one incoming edge, one outgoing edge, a single cycle or non-self-intersection.

**Parameters / compute:** approximately `0.04 M` parameters; `O(M²D)` and negligible at `M=40`.

### 3.7 Polygon Assembly and Output Projection

After topology reconciliation, let `π:[K]` be the single-cycle order over valid slots and collect center vertices

`V_roi=[p_{π_1},…,p_{π_K}]∈[-0.25,1.25]^{K×2}`.

Map them from the selected outer RoI `R=(x_R,y_R,w_R,h_R)` to the augmented image:

`V_img[:,0]=x_R+w_R V_roi[:,0],  V_img[:,1]=y_R+h_R V_roi[:,1].`

Undo resize, crop, flip and padding using the sample metadata, clip to the original image, and emit `V_orig:[K,2]`. Only the center `p` becomes a polygon vertex; `p^-` and `p^+` are local direction supervision. Rasterize the closed polygon only after cycle validation to obtain the evaluation mask/RLE.

Expose topology quality as the diagnostic scalar

`q_topo=exp[K^{-1}Σ_m log P_{π_m,π_{m+1}}]`,

and set it to zero when cycle validation fails. Keep the inherited detector confidence as the instance score in v0.1; validity and `q_topo` are structural diagnostics, not uncalibrated score multipliers. Instances with `K<3`, non-finite coordinates or a failed single-cycle/non-intersection check are rejected and counted.

### 3.8 Losses and Training Routing

All terms are normalized by their own valid element count before weighting:

`L = L_det + L_contour + L_x0 + 2L_sbox + L_valid + 5L_prim^raw + L_topo.`

| Loss | Definition | Gradient destination |
|---|---|---|
| `L_det` | inherited FCOS losses | backbone, neck, detector |
| `L_contour` | two-class pixel CE on `[N,2,32,32]` | mask head, DFE, SFE, shared image features |
| `L_x0` | weighted SmoothL1 in standardized state; null weight 0.1 | denoiser and state embeddings |
| `L_sbox` | active-support L1 + GIoU in decoded box space | denoiser |
| `L_valid` | sigmoid focal, `α=.25,γ=2` | validity head, local fusion, conditioner and denoiser |
| `L_prim^raw` | V3 reversal-invariant primitive loss: 6D SmoothL1 (`β=.01`) + center SmoothL1 + incident-direction cosine terms | primitive head, final grid-sample path, conditioner and denoiser |
| `L_topo` | bidirectional Sinkhorn successor NLL | topology Q/K, geometry-bias MLP, primitive head and upstream slot features |

For each of the forward and reversed GT triples, compute

`L_prim^r=L_coord^r+0.5L_center^r+0.5L_angle^r`,

where `L_coord` is mean 6D SmoothL1, `L_center` is mean 2D SmoothL1 on `p`, and

`L_angle=0.5[(1-cos(d^-,d_gt^-))+(1-cos(d^+,d_gt^+))]`

uses epsilon-normalized incident directions; mask an angle term when the corresponding GT incident edge is shorter than `10^-4`. Then `L_prim^raw=min_r L_prim^r`. This must be a new explicit loss: the current repository's `LineLoss` implements reversal-invariant coordinate and center terms, but its configured `with_angle_loss=True` flag is not consumed in `forward()`. Keep the external coefficient `5` for scale continuity; use the unchanged existing loss only for the exact V1/V2 reproduction gate.

`L_prim^raw` and `L_sbox` are evaluated only on clean-coupled active slots. Null slots have no primitive target; they are supervised by `L_valid` and the `0.1×` null term in `L_x0` only.

No raster Dice, Chamfer, soft self-intersection or extra order CE is included in v0.1. These terms overlap existing supervision or introduce difficult gradients before the basic hypothesis is tested.

#### Building-box exposure-gap training

First expand every base building box by the inherited context factor `1.1`. Every target and CFE crop then uses the same selected outer RoI:

- 45% clean GT building boxes;
- 35% moderate jitter: center up to `±10%` of width/height and log-scale/aspect up to `±15%`;
- 10% hard jitter: up to `±20%`;
- 10% stop-gradient, GT-matched FCOS proposals after detector warm-up; use hard jitter before proposals are reliable.

Support targets and primitive coordinates are re-normalized inside that perturbed RoI. Redraw a synthetic jitter, or skip a matched proposal for generator supervision, if any GT primitive coordinate leaves `[-0.25,1.25]`; report this rejection rate. Detector proposals are detached after matching/NMS, and the detector still learns only from clean GT via `L_det`.

#### Recommended training phases

1. **V2 reverse-reproduction gate:** reshape each `E_l:[N,64,256]` to `[64,N,256]`, pass `(E_4,E_3,E_2)` and their separately reshaped positional encodings into the unchanged V1 primitive/order decoder, and retain `Q=30/order=36`. Reproduce the direction of Table V: baseline < SFE-only/DFE-only < SFE+DFE. Do not proceed if the conditioner cannot beat V1.
2. **Deterministic controls:** replace the V1 decoder with the same support denoiser/heads, initialize from clean anchors `z(A)` with no Gaussian noise, and test one call plus a shared-weight two-call recurrent refiner. This matches both parameters and, for the two-call control, NFE.
3. **Diffusion enablement:** train random `t` with `L_x0/L_sbox`; evaluate 1/2/4 NFE.
4. **Topology enablement:** add successor head only after geometry/validity stabilize.
5. **Joint fine-tuning:** unfreeze backbone/neck at reduced LR; retain direct `L_contour` so the mask branch cannot drift.

---

## 4. Parameter and Compute Budget

### 4.1 Parameter budget

The current local R50 P2PFormer V1 build contains `41.275 M` parameters:

| Component | Params |
|---|---:|
| ResNet-50 backbone | 24.379 M |
| multiscale neck | 3.839 M |
| FCOS head | 4.863 M |
| current V1 line/primitive/order head | 8.194 M |
| **V1 total** | **41.275 M** |

V3 removes the current `8.194 M` line head and installs:

| V3 component | Approx. params |
|---|---:|
| V2-compatible contour conditioner | 0.55 M |
| state/anchor/time embeddings | 0.23 M |
| 3 denoising blocks | 4.92 M |
| clean-state output head | 0.001 M |
| final local fusion + primitive/validity heads | 0.14 M |
| topology head | 0.04 M |
| **new contour/generation subsystem** | **5.88 M** |
| inherited scene subsystem | 33.082 M |
| **estimated P2PFormerV3 total** | **≈38.96 M** |

The proposed model is approximately `5.6%` smaller than the current V1 build; there is no unexplained parameter inflation.

### 4.2 Compute budget

Because compute scales with detected instance count, report:

`MAC_total = MAC_shared-image + N·MAC_per-RoI.`

For `[1,3,640,640]`, the repository's stock FLOPs command fails because `forward_dummy` passes the neck's nested `(line_feature,bbox_features)` output directly to FCOS. A read-only corrected forward using the actual `P3→P5` route reports `90.35 G` hook-counted multiply operations for backbone + neck + FCOS. Treat this as a **shared-image lower bound**: MMCV's counter includes Conv/Linear/norm/activation children but does not price the custom DCNv2 or multi-scale deformable sampling kernels completely. This shared term is identical for V1 and V3 and therefore cancels in the incremental comparison.

Approximate per-RoI costs:

| Component | 2 NFE | 4 NFE |
|---|---:|---:|
| V2 conditioner, once | 0.70 GMAC | 0.70 GMAC |
| condition K/V cache, once | 0.025 GMAC | 0.025 GMAC |
| set denoiser | 0.236 GMAC | 0.472 GMAC |
| final primitive + topology | <0.01 GMAC | <0.01 GMAC |
| **per RoI** | **≈0.97 GMAC** | **≈1.21 GMAC** |

If one multiply-add is counted as two FLOPs, these are approximately `1.94` and `2.42 GFLOPs/RoI`. At ten detected buildings, the hook-counted lower-bound totals are approximately `100.05/102.45 G` operations, of which V3 adds `9.7/12.1 GMAC` beyond the shared image path. Exact latency must be measured because deformable sampling, RoIAlign, grid sampling and small attention kernels are not predicted well by FLOPs alone.

### 4.3 Activation memory

- Three RoI crops: `3×[N,256,32,32]` ≈ `1.5 MiB/RoI` in FP16.
- Largest DFE attention score per processed scale: `[N,8,1024,64]` ≈ `1.0 MiB/RoI` in FP16.
- Maximum `33×33` one-channel soft-dilation unfold is ≈`2.13 MiB/RoI` in FP16 and must be released before processing the next feature level.
- Denoiser attention scores are <`0.1 MiB/RoI`.
- Process P2/P3/P4 sequentially and chunk RoIs (default chunk 16) to bound peak memory.

---

## 5. Routing & Gradient Flow Audit

### 5.1 Routing audit

| Route | Information carried | Rejoin | Preservation / loss |
|---|---|---|---|
| scene → detector | semantic multiscale object evidence | independent detector loss | compressive to boxes |
| scene → CFE | high-resolution boundary evidence | SFE/DFE fusion | spatial structure retained in `32×32` crop |
| SFE branch | where informative contour samples lie | concat with masked dense samples | sparse/compressive, positions explicitly retained |
| DFE branch | dense contour context and mask logits | same-position resampling | dense map retained until mask/fusion |
| unmasked residual in DFE | original crop evidence | residual add | information-preserving safety path |
| Feature Mask | contour-band relevance | multiplicative gate with `0.1` floor | suppressive but never fully destructive |
| support stream | noisy geometry and spatial anchor | self-attention residual | preserves slot identity/state |
| condition stream | cached P4/P3/P2 contour evidence | cross-attention residual | condition is read-only; no recurrent drift |
| final local grid | evidence inside predicted clean support | additive projection into `H_3` | local spatial layout compressed by GAP |
| topology solver | discrete cycle legality | final polygon assembly | non-differentiable, intentionally outside geometry training |

The highest-risk reshape is `[N,256,8,8]→[N,64,256]`. It is information-preserving; deformed `(x,y)` positions are carried separately through `PE_l`. Values do not receive positional addition, avoiding corruption of visual content.

### 5.2 Gradient audit

- Backbone/neck receive `L_det` and all contour/generator losses after joint unfreezing.
- Outer building RoIs are stop-gradient inputs: GT/jitter boxes are data, while detector selection/NMS is discrete and the inherited RoIAlign path does not differentiate box coordinates. FCOS therefore learns from `L_det`; the explicit RoI-jitter curriculum closes the generator exposure gap.
- SFE offsets receive gradients through both sparse sampling and same-position masked-dense sampling.
- DFE and mask head receive direct `L_contour` plus indirect primitive/diffusion gradients through the soft residual gate.
- The `0.1` mask floor prevents a low-quality mask from zeroing both forward information and backward gradients.
- Pre-norm residuals give every denoiser block an identity gradient path.
- AdaLN gates start at zero, so recursive denoising begins as a near-identity perturbation rather than an unstable deep transformation.
- `L_x0/L_sbox` supervise all three denoising blocks only through the final head; three blocks are shallow enough that block-level deep supervision is unnecessary.
- The explicit final `grid_sample` lets `L_prim` update predicted clean support geometry through differentiable box coordinates while it remains inside the sampling range.
- The Hungarian/subtour/2-opt inference solver does not backpropagate. `L_topo` is therefore mandatory; without it the topology features would be a dead branch.

Required training diagnostics:

- SFE offset norm, tanh saturation and out-of-bound rate;
- Feature Mask positive-area ratio and hard/soft IoU;
- per-branch gradient norms for SFE, DFE, mask, denoiser and topology;
- active/null validity calibration;
- duplicate supports, topology subtours and 2-opt counts;
- quality at 1/2/4 NFE.

---

## 6. Key Design Decisions

1. **Diffuse 4D AABBs, not final vertices or rotated boxes.** This isolates local evidence generation and avoids cyclic angle and polygon permutation ambiguities.
2. **Use a fixed clean anchor coupling across timesteps.** It prevents slot switching while preserving set-order equivariance when anchor-state pairs are jointly permuted.
3. **Reconstruct full V2 semantics, but use a soft residual Feature Mask.** The contour band suppresses invalid evidence without letting a false-negative mask erase weak boundaries.
4. **No noisy-support RoIAlign inside DDIM.** Noisy geometry is Q; cached contour evidence is K/V. Local sampling occurs only after the support is clean.
5. **Replace order bins with a successor cycle.** One-in/one-out matching plus subtour merging and 2-opt makes topology an explicit invariant.
6. **Use 2 NFE by default.** Four steps are justified only if they form a measurable quality Pareto over the one-step baseline.

---

## 7. What Would Invalidate This Design

1. **V2 reverse reproduction fails.** If SFE+DFE+fusion cannot reproduce the qualitative Table V ordering, V3 condition features are untrusted and diffusion results are uninterpretable.
2. **One- or two-call deterministic control matches 2/4 diffusion steps.** Then Gaussian diffusion is unnecessary; retain support boxes and topology but use the deterministic refiner.
3. **Support targets are unstable.** If small annotation perturbations cause large AABB changes or support recall does not correlate with primitive recall, diffuse primitives directly instead.
4. **Anchor collisions dominate.** If many valid corners repeatedly match the same spatial neighborhood or active recall saturates below the M=40 capacity ceiling, move to M=64 or proposal-guided set coupling.
5. **Feature Mask false negatives erase cross-domain contours.** If the soft gate still hurts Test2/weak-boundary subsets, provide the unmasked crop as an explicit second K/V memory or remove multiplicative masking.
6. **Fixed 8-pixel semantics do not transfer across GSD.** Replace pixel width with GSD-aware or RoI-relative width only after showing the domain dependence.
7. **Topology remains mostly postprocessing.** If subtour merging/2-opt frequently changes predictions, the successor head has not learned a single cycle and needs stronger graph supervision.
8. **Per-RoI cost is unacceptable in dense scenes.** If CFE dominates latency, keep SFE and remove the dense cross-attention branch before shrinking the denoiser; V2 ablations show SFE is the stronger individual branch.
9. **Detector recall remains the upper bound.** If missed building instances dominate error, primitive support diffusion cannot solve the problem; scene-level localization must become a separate research phase.

---

## 8. Minimum Ablation Matrix

| Axis | Required settings |
|---|---|
| V2 reconstruction | V1 / SFE / DFE / SFE+DFE |
| Feature Mask | oracle band / predicted hard / predicted soft residual / no mask |
| outer RoI | GT / moderate jitter / hard jitter / detector box |
| support representation | point-only / 4D AABB / rotated box only if 4D wins |
| generator | one-step deterministic / two-call deterministic recurrent / diffusion 1 NFE / 2 NFE / 4 NFE |
| capacity | M=30 / M=40 |
| coupling | per-step Hungarian / fixed clean anchor coupling |
| topology | V1 36-bin order / continuous sort / successor-cycle head |
| condition | P2 only / P2–P4 native memories |

Primary paper claim is supported only if:

- 2/4-step generation beats the one-step same-capacity baseline on difficult subsets;
- soft-mask conditioning improves over no-mask and hard-mask variants;
- successor topology reduces invalid polygons without lowering AP;
- gains remain after matching parameters and reporting latency.

---

## 9. Three Questions for the Next Design Review

1. **The load-bearing representation choice is 4D AABB support.** Do you agree that rotated support boxes should remain an ablation rather than enter v0.1, and what failure evidence would justify adding angle?
2. **The load-bearing V2 inheritance choice is soft residual masking.** Should the first milestone prioritize a faithful hard-mask-semantics Table V reproduction, or move directly to the safer soft-mask V3 conditioner after reproducing SFE alone?
3. **The load-bearing compute choice is 2 NFE.** What GPU/latency ceiling should define success, and is a 4-step quality setting acceptable if it adds about `0.24 GMAC/RoI` over 2 steps?

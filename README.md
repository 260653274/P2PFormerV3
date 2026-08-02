<div align="center">

# [P2PFormer: A Primitive-to-polygon Method for Regular Building Contour Extraction from Remote Sensing Images](https://arxiv.org/pdf/2406.02930)
[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), Shiqing Wei, Yikang Zhou, Muying Luo, Wenling Yu, [ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN)
</div>

## Install

The tested Blackwell runtime uses Python 3.11, PyTorch 2.8 with CUDA 12.8,
and `mmcv-full==1.7.2` compiled for compute capability 12.0.

```shell
# create the conda environment
conda create -n p2pformer python=3.11 pip -y
conda activate p2pformer

# install PyTorch for CUDA 12.8
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# install build dependencies, then compile MMCV CUDA operators for Blackwell
python -m pip install \
  "pip<25.3" "setuptools<81" wheel packaging ninja "cython>=3.0,<3.1"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=4
python -m pip install \
  --no-build-isolation --no-cache-dir --no-deps \
  mmcv-full==1.7.2

# install the remaining runtime dependencies and P2PFormer
python -m pip install -r requirements-p2pformer-runtime.txt
python -m pip install --no-build-isolation --no-deps -e .

# verify the imported legacy APIs and a real CUDA NMS operation
python tools/check_mmcv_compatibility.py
```

Do not install the MMCV 2.x packages `mmcv` or `mmcv-lite` in this
environment. This branch embeds MMDetection 2.25.1 and depends on MMCV 1.x
APIs removed in MMCV 2.x. See [the compatibility note](docs/en/compatibility.md#p2pformer-runtime-boundary).

## Prepare datas
Download the WHU, WHU-Mix and CrowdAI datastes, then change the data path in [whu_line.py](./p2pformer/configs/_base_/datasets/whu_line.py), [whu-mix_line.py](./p2pformer/configs/_base_/datasets/whu-mix_line.py) and [crowdAI_line](./p2pformer/configs/_base_/datasets/crowdAI_line.py).

## Model Zoo
The pretrained weights are available at [here](https://huggingface.co/zhangtao-whu/P2PFormer/tree/main).

### Reproduce the ResNet-50 WHU-Mix results

The released `p2pformer_corner_r50_whu_mix.pth` checkpoint corresponds to
`p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py`. After activating
the `p2pformer` Conda environment, evaluate both paper test splits with:

```shell
bash tools/reproduce_whumix_r50.sh
```

The script verifies the official checkpoint SHA-256 before inference and writes
the raw predictions and evaluation artifacts under `work_dirs/reproduction/`.
Use `test1` or `test2` as the second argument to run only one split:

```shell
bash tools/reproduce_whumix_r50.sh \
  checkpoints/p2pformer_corner_r50_whu_mix.pth test1
```

The paper reports mask AP/AP50/AP75 of 60.6/87.3/68.9 on test1 and
50.7/79.9/54.4 on test2. On an RTX 5060 Ti with PyTorch 2.8.0+cu128 and the
required `mmcv-full==1.7.2`, the released checkpoint reproduces
60.3/87.0/68.6 and 49.9/79.0/53.3, respectively. The checkpoint was originally
produced with PyTorch 1.13.1+cu117 on A800 GPUs, so exact bitwise agreement is
not expected across these CUDA operator generations.

## Getting Started

### train 

```shell
# on whu dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/configs/p2pformer_corner_whu.py 8
# on whu-mix dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py 8
# on crowdai dataset
PYTHONPATH=. bash tools/dist_train.sh p2pformer/configs/configs/p2pformer_corner_r50_crowdai.py 8
```

### test
```shell
# on whu dataset
PYTHONPATH=. bash tools/dist_test.sh p2pformer/configs/configs/p2pformer_corner_whu.py /path/to/model.pth 8 --eval segm
# on whu-mix dataset
PYTHONPATH=. bash tools/dist_test.sh p2pformer/configs/configs/p2pformer_corner_r50_whu-mix.py /path/to/model.pth 8 --eval segm
# on crowdai dataset
PYTHONPATH=. bash tools/dist_test.sh p2pformer/configs/configs/p2pformer_corner_r50_crowdai.py /path/to/model.pth 8 --eval segm
```

### visualization
1. Please uncomment lines 229-231 in [p2pformer.py](./p2pformer/models/p2pformer.py), and the polygon prediction results will be stored in `work_dirs/json_preds/`.
2. Run the test script or demo script.
3. Run the [polygon_show.py](./tools/polygon_visualize/polygon_show.py) to obtain the visualization results.


```BibTeX
@article{zhang2024p2pformer,
  title={P2PFormer: A Primitive-to-polygon Method for Regular Building Contour Extraction from Remote Sensing Images},
  author={Zhang, Tao and Wei, Shiqing and Zhou, Yikang and Luo, Muying and Yu, Wenling and Ji, Shunping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```

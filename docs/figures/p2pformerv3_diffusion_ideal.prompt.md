# Prompt: P2PFormerV3 Idealized Diffusion Figure

- Mode: built-in `image_gen`
- Use case: `scientific-educational`
- Input image role: composition and visual-language reference only; not an edit target

```text
Use case: scientific-educational
Asset type: publication-ready computer-vision paper figure illustrating the ideal behavior of P2PFormerV3
Input images: Image 1 is a composition and visual-language reference only; do not copy its exact building or overlays.
Primary request: create a clean four-panel horizontal scientific figure showing conditional diffusion of primitive-level geometric support boxes for regular building contour extraction.
Scene/backdrop: the exact same top-down high-resolution aerial crop of one light-gray L-shaped industrial building must be repeated identically in all four panels; white page background, equal square panels, thin dark-gray borders, generous spacing.
Panel progression:
(a) at t = T, many small translucent magenta and violet rotated rectangular support boxes are randomly scattered, misaligned, duplicated, and partly outside the building;
(b) at an intermediate noisy step, invalid boxes fade and the remaining boxes move toward the roof boundary, but some are still misaligned;
(c) near t = 0, a sparse set of cyan and blue support boxes tightly aligns with roof corners and edge segments, with consistent orientation and minimal duplicates;
(d) final result: only valid boundary-aligned support boxes remain faintly visible, crisp orange corner primitives connect into one closed non-self-intersecting polygon exactly tracing the building footprint, with a subtle transparent blue interior mask.
Add simple right-pointing arrows between panels to communicate denoising. Under each panel, render only these exact labels, once each: "(a)  t = T", "(b)  Denoising", "(c)  t → 0", "(d)  Final polygon".
Style/medium: precise academic remote-sensing visualization, realistic orthophoto base with clean vector-like overlays, similar clarity and restrained visual density to Image 1.
Color palette: grayscale aerial image; noisy supports magenta/violet; converged supports cyan/blue; final polygon orange; white background.
Scientific constraints: support boxes must be small local rectangles around corner or edge primitives, never large building detection boxes; the final orange polygon must form one valid closed contour with no self-intersection; show progressive correspondence across stages; preserve the identical building geometry, crop, scale, and camera angle in every panel.
Constraints: landscape figure, high legibility at paper-column scale, no title, no caption paragraph, no axes, no legend, no logos, no watermark, no decorative icons, no extra text.
```

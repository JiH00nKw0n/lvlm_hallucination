# k=32 Full Metrics (6 methods, excluding VL-SAE)

CLIP ViT-B/32. COCO-trained, L=8192 (SharedEnc L=4096). k=32, 30 ep.

## Table A — Reconstruction (COCO test vs ImageNet val)

| Method | COCO recon | COCO img | COCO txt | **ImageNet val recon** | IN img | IN txt | Transfer gap |
|---|---|---|---|---|---|---|---|
| Shared | 0.0716 | 0.0946 | 0.0486 | **0.1555** | 0.1416 | 0.1694 | +0.0840 |
| Separated | 0.0701 | 0.0934 | 0.0468 | **0.1529** | 0.1392 | 0.1666 | +0.0828 |
| IsoAlign | 0.0710 | 0.0941 | 0.0479 | **0.1551** | 0.1400 | 0.1703 | +0.0841 |
| GrpSparse | 0.0779 | 0.1046 | 0.0512 | **0.1691** | 0.1559 | 0.1822 | +0.0912 |
| Ours | 0.0701 | 0.0934 | 0.0468 | **0.1529** | 0.1392 | 0.1666 | +0.0828 |
| SharedEnc | 0.0873 | 0.1136 | 0.0610 | **0.1857** | 0.1609 | 0.2104 | +0.0984 |

## Table B — COCO retrieval (pessimistic tie)

| Method | T2I R@1 | R@5 | R@10 | T2I tie | I2T R@1 | R@5 | R@10 | I2T tie |
|---|---|---|---|---|---|---|---|---|
| Shared | 2.25% | 6.57% | 9.45% | 3218 | 1.00% | 5.68% | 9.48% | 10504 |
| Separated | 0.01% | 0.12% | 0.25% | 3630 | 0.04% | 0.08% | 0.16% | 14550 |
| IsoAlign | 2.35% | 5.73% | 8.02% | 3178 | 1.88% | 5.56% | 8.78% | 11101 |
| GrpSparse | 3.45% | 8.71% | 12.03% | 3353 | 5.16% | 11.52% | 15.74% | 11833 |
| Ours | 6.68% | 18.36% | 26.08% | 1 | 5.26% | 13.58% | 19.08% | 1 |
| SharedEnc | 10.32% | 24.74% | 33.55% | 95 | 18.48% | 36.86% | 46.52% | 5 |

## Table C — Alive latents (COCO 50k paired subset)

| Method | img alive/L | txt alive/L | img % | txt % |
|---|---|---|---|---|
| Shared | 5113/8192 | 3267/8192 | 62.4% | 39.9% |
| Separated | 2989/4096 | 2775/4096 | 73.0% | 67.7% |
| IsoAlign | 4996/8192 | 3428/8192 | 61.0% | 41.8% |
| GrpSparse | 7142/8192 | 4584/8192 | 87.2% | 56.0% |
| Ours | 2989/4096 | 2775/4096 | 73.0% | 67.7% |
| SharedEnc | 3114/4096 | 3428/4096 | 76.0% | 83.7% |

## Table D — Cross COCO→ImageNet val (50k images, 1000 classes)

| Method | Linprobe (masked+top1, t=0.1) | **Zero-shot (raw, all slots)** | Zero-shot (dominant-masked, dense) | Zero-shot (masked+top1, t=0.1) | Dominant slots |
|---|---|---|---|---|---|
| Shared | 12.94% | **15.71%** | 15.71% | 0.26% | 8 |
| Separated | 12.01% | **0.15%** | 0.15% | 0.10% | 5 |
| IsoAlign | 12.73% | **13.77%** | 13.77% | 0.14% | 6 |
| GrpSparse | 28.75% | **17.22%** | 17.22% | 2.81% | 0 |
| Ours | 12.00% | **13.07%** | 19.17% | 0.45% | 5 |
| SharedEnc | 16.90% | **26.79%** | 27.34% | 3.75% | 2 |

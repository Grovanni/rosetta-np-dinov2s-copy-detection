# Third-party notices

## DINOv2

Rosetta NP uses the DINOv2 ViT-S/14 architecture and is initialized from `facebook/dinov2-small`.

- Repository: https://github.com/facebookresearch/dinov2
- Paper: *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., 2023), https://arxiv.org/abs/2304.07193
- License: Apache License 2.0 for the DINOv2 code and model weights, as stated by the upstream project.

## DISC21 / Image Similarity Challenge

DISC21 data and benchmark structure are described by:

- Douze et al., *The 2021 Image Similarity Dataset and Challenge*, https://arxiv.org/abs/2106.09672
- Yokoo et al., *Results and Findings of the 2021 Image Similarity Challenge*, https://arxiv.org/abs/2202.04007

DISC21 images are not redistributed in this repository. Users must obtain datasets through their official distribution and comply with their terms.

## SSCD

The published comparison uses the official SSCD ResNet-50 `disc_mixup` global descriptor:

- Repository: https://github.com/facebookresearch/sscd-copy-detection
- Paper: *A Self-Supervised Descriptor for Image Copy Detection* (Pizzi et al., 2022), https://arxiv.org/abs/2202.10261

SSCD weights are not redistributed here.

## Python dependencies

This package depends on PyTorch, Transformers, safetensors, Pillow and NumPy. Each dependency remains under its own license.


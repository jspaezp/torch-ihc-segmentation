
# IHC segmentation

# Motivation

It is originally an in-house utility to quantify large IHC images.

## Milestones

1. MVP
    - Model definition using torchvision as a base
    - Model inference returning per patch areas of empty/cell/tissue
        - Tiling on large images
    - Training script using CLI in a notebook
    - Data generation
    - Basic mask visualization
2. Deployment additions
    - Torchscript export
    
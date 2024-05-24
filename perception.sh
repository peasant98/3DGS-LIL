#!/bin/bash

cd data/real-world/lab-scene
unzip database.zip

cd ../../..

cd LEGaussians

conda create -n legaussians python=3.8 -y
conda activate legaussians

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install tqdm plyfile timm open_clip_torch scipy six configargparse pysocks python-dateutil imageio seaborn opencv-python scikit-learn tensorboard Pillow==9.5.0

# Install local packages from the 'submodules' directory
# a modified gaussian splatting (+ semantic features rendering)
cd submodules/diff-gaussian-rasterization/ && python -m pip install -e . && cd ../..
# simple-knn
cd submodules/simple-knn/ && python -m pip install -e . && cd ../..
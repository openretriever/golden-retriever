#!/bin/bash

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth # SwinT model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth # SwinB model
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth

mkdir -p src/models/mapping/utils_d3fields/ckpts
mv sam_vit_h_4b8939.pth src/models/mapping/utils_d3fields/ckpts/
mv groundingdino_swint_ogc.pth src/models/mapping/utils_d3fields/ckpts/
mv groundingdino_swinb_cogcoor.pth src/models/mapping/utils_d3fields/ckpts/
mkdir -p external/XMem/saves
mv XMem.pth external/XMem/saves/

# # Download DON (optional)
# wget https://data.csail.mit.edu/labelfusion/pdccompressed/trained_models/stable/shoes_consistent_M_background_0.500_3.tar.gz
# tar -xf shoes_consistent_M_background_0.500_3.tar.gz
# mv shoes_consistent_M_background_0.500_3 don/
# mv don/ ckpts/
# rm shoes_consistent_M_background_0.500_3.tar.gz

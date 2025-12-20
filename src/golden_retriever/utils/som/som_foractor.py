# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import os
import pathlib
import warnings

import matplotlib.colors as mcolors
import torch
from detectron2.data import MetadataCatalog
from openai import OpenAI
from PIL import Image
from seem.modeling import build_model as build_model_seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from segment_anything import sam_model_registry
from semantic_sam import build_model
from semantic_sam.BaseModel import BaseModel
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES

from retriever.utils.som.gpt4v import request_gpt4v

from .task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from .task_adapter.seem.tasks import (
    inference_seem_pano,
)
from .task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto

warnings.filterwarnings("ignore")
metadata = MetadataCatalog.get("coco_2017_train_panoptic")

css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

client = OpenAI()

"""
build args
"""
root_dir = str(pathlib.Path.cwd()) + "/"
assets_root = os.path.join(root_dir, "src/utils/som/")
semsam_cfg = assets_root + "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = assets_root + "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = assets_root + "/download_data/swinl_only_sam_many2many.pth"
sam_ckpt = assets_root + "/download_data/sam_vit_h_4b8939.pth"
seem_ckpt = assets_root + "/download_data/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
# opt_seem = init_distributed_seem(opt_seem)


"""
build model
"""
model_semsam = (
    BaseModel(opt_semsam, build_model(opt_semsam))
    .from_pretrained(semsam_ckpt)
    .eval()
    .cuda()
)
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = (
    BaseModel_Seem(opt_seem, build_model_seem(opt_seem))
    .from_pretrained(seem_ckpt)
    .eval()
    .cuda()
)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
        )

history_images = []
history_masks = []
history_texts = []


@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    global history_images
    history_images = []
    global history_masks
    history_masks = []
    if slider < 1.5:
        model_name = "seem"
    elif slider > 2.5:
        model_name = "sam"
    else:
        if mode == "Automatic":
            model_name = "semantic-sam"
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = "sam"

    if label_mode == "Alphabet":
        label_mode = "a"
    else:
        label_mode = "1"

    text_size, hole_scale, island_scale = 640, 100, 100
    text, text_part, text_thresh = "", "", "0.0"
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        semantic = False

        # if mode == "Interactive":
        #     labeled_array, num_features = label(np.asarray(image['mask'].convert('L')))
        #     spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == "semantic-sam":
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(
                model,
                image,
                level,
                text,
                text_part,
                text_thresh,
                text_size,
                hole_scale,
                island_scale,
                semantic,
                label_mode=label_mode,
                alpha=alpha,
                anno_mode=anno_mode,
                *args,
                **kwargs,
            )

        elif model_name == "sam":
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(
                    model, image, text_size, label_mode, alpha, anno_mode
                )
            elif mode == "Interactive":
                raise NotImplementedError("Removed unused mode")

        elif model_name == "seem":
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(
                    model, image, text_size, label_mode, alpha, anno_mode
                )
            elif mode == "Interactive":
                raise NotImplementedError("Removed unused mode")

        # convert output to PIL image
        history_masks.append(mask)
        history_images.append(Image.fromarray(output))
        return output, []


def gpt4v_response(message, history):
    global history_images
    global history_texts
    history_texts = []
    try:
        res = request_gpt4v(message, history_images[0])
        history_texts.append(res)
        return res
    except Exception:
        return None

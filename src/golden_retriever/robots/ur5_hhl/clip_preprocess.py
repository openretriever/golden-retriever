import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from retriever.skill_training.gem.cliport.tasks import cameras
from retriever.skill_training.gem.cliport.utils import utils
from retriever.skill_training.gem.lepp.parser import parse_instruction


def clip_preprocess(n_px):
    # Originally from CLIP in Mingxi's code: .clip_revised.clip import preprocess as clip_preprocess
    from torchvision.transforms import CenterCrop, Compose, Normalize, Resize

    try:
        from torchvision.transforms import InterpolationMode

        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC, antialias=True),
            CenterCrop(n_px),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class CLIPFeatureProcessor:
    def __init__(self, clip_type="normal"):
        # set clip
        model_id = "openai/clip-vit-base-patch32"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tmodel = CLIPTextModelWithProjection.from_pretrained(model_id).to(
            self.device
        )
        self.vmodel = CLIPVisionModelWithProjection.from_pretrained(model_id).to(
            self.device
        )

        self.clip_type = clip_type

        self.model.eval()
        self.tmodel.eval()

    def get_clip_text_feature(self, lan_goal):
        # color_list = ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan', 'orange', 'purple', 'pink', 'white']
        # for color in color_list:
        #     if color in lan_goal:
        #         lan_goal = lan_goal.replace(color, '')
        with torch.no_grad():
            text = self.tokenizer([lan_goal], padding=True, return_tensors="pt").to(
                self.device
            )
            text_features = self.tmodel(**text).text_embeds
        return text_features.cpu().numpy()

    def get_clip_feature(
        self, img, lan_goal, kernel_size=40, stride=20, normalization=True
    ):
        with torch.no_grad():
            rgb_raw = torch.tensor(img).permute((2, 0, 1)).unsqueeze(0).to(self.device)
            pad_size = kernel_size // 2
            rgb = F.pad(
                input=rgb_raw,
                pad=(pad_size, pad_size, pad_size, pad_size),
                mode="constant",
            )

            patches = rgb.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(
                patches.shape[0] * patches.shape[1], -1, kernel_size, kernel_size
            )
            patches = patches.permute((1, 0, 2, 3))

            if self.clip_type == "c4max":
                patches_rot_list = []
                for angle in range(3):
                    patches_rot_list.append(torch.rot90(patches, angle, [2, 3]))

                patches_rot = torch.cat([patches, *patches_rot_list], axis=0)

                processor = clip_preprocess(224)
                patches_rot = processor(patches_rot / 255.0)

                text = self.tokenizer([lan_goal], padding=True, return_tensors="pt").to(
                    self.device
                )
                text_features = self.tmodel(**text).text_embeds

                # patches = self.processor(
                #     images=patches,  # big patch image sent to CLIP
                #     return_tensors="pt",  # tell CLIP to return pytorch tensor
                # ).to(device).pixel_values  # too slow

                # score = self.model(**inputs)
                score = self.model(input_ids=text.input_ids, pixel_values=patches_rot)
                scores = score.logits_per_image
                scores = scores.reshape([4, -1, 1]).max(axis=0).values
                scores = scores.reshape(rgb.shape[0], patches.shape[0])

            elif self.clip_type == "normal":
                processor = clip_preprocess(224)
                patches = processor(patches / 255.0)

                text = self.tokenizer([lan_goal], padding=True, return_tensors="pt").to(
                    self.device
                )
                text_features = self.tmodel(**text).text_embeds

                # patches = self.processor(
                #     images=patches,  # big patch image sent to CLIP
                #     return_tensors="pt",  # tell CLIP to return pytorch tensor
                # ).to(device).pixel_values  # too slow

                # score = self.model(**inputs)
                score = self.model(input_ids=text.input_ids, pixel_values=patches)
                scores = score.logits_per_image
                scores = scores.reshape(rgb.shape[0], patches.shape[0])
            # clip the scores
            scores = torch.clip(scores - scores.mean(dim=-1), 0, torch.inf)

            # normalize scores
            if normalization:
                scores = (scores - scores.min(dim=-1).values) / (
                    scores.max(dim=-1).values - scores.min(dim=-1).values
                )

            clip_feature = scores.reshape(
                rgb.shape[0],
                1,
                int((rgb_raw.shape[2]) / stride + 1),
                int((rgb_raw.shape[3]) / stride + 1),
            )
            clip_feature = nn.functional.interpolate(
                clip_feature,
                size=[rgb_raw.shape[2], rgb_raw.shape[3]],
                mode="bilinear",
                align_corners=True,
            )

            clip_feature = clip_feature.squeeze().unsqueeze(-1)
            del patches, rgb_raw
            # torch.cuda.empty_cache()
        return clip_feature.cpu().numpy(), text_features.cpu().numpy()

    def get_clip_feature_from_image(
        self, img, img_goal, kernel_size=40, stride=20, normalization=True
    ):
        with torch.no_grad():
            rgb_raw = torch.tensor(img).permute((2, 0, 1)).unsqueeze(0).to(self.device)
            pad_size = kernel_size // 2
            rgb = F.pad(
                input=rgb_raw,
                pad=(pad_size, pad_size, pad_size, pad_size),
                mode="constant",
            )

            patches = rgb.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(
                patches.shape[0] * patches.shape[1], -1, kernel_size, kernel_size
            )
            patches = patches.permute((1, 0, 2, 3))

            img_goal = (
                torch.from_numpy(img_goal)
                .permute((2, 0, 1))
                .unsqueeze(0)
                .to(self.device)
            )

            processor = clip_preprocess(224)
            patches = processor(patches / 255.0)
            img_goal = processor(img_goal / 255.0)

            # score = self.model(**inputs)
            outputs = self.vmodel(pixel_values=patches)
            image_embeds = outputs["image_embeds"]

            goal_outputs = self.vmodel(pixel_values=img_goal)
            goal_image_embeds = goal_outputs["image_embeds"]

            scores = torch.from_numpy(
                cosine_similarity(
                    image_embeds.cpu().numpy(), goal_image_embeds.cpu().numpy()
                )
            )
            scores = scores.reshape(rgb.shape[0], patches.shape[0])
            # clip the scores
            scores = torch.clip(scores - scores.mean(dim=-1), 0, torch.inf)

            # normalize scores
            if normalization:
                scores = (scores - scores.min(dim=-1).values) / (
                    scores.max(dim=-1).values - scores.min(dim=-1).values
                )

            clip_feature = scores.reshape(
                rgb.shape[0],
                1,
                int((rgb_raw.shape[2]) / stride + 1),
                int((rgb_raw.shape[3]) / stride + 1),
            )
            clip_feature = nn.functional.interpolate(
                clip_feature,
                size=[rgb_raw.shape[2], rgb_raw.shape[3]],
                mode="bilinear",
                align_corners=True,
            )

            clip_feature = clip_feature.squeeze().unsqueeze(-1).numpy()
            del patches, rgb_raw
            # torch.cuda.empty_cache()
        return clip_feature, goal_image_embeds

    def get_clip_feature_from_text_and_image(
        self, img, lang_goal, img_goal, kernel_size=40, stride=20
    ):
        clip_feature_text, text_emb = self.get_clip_feature(
            img, lang_goal, kernel_size=kernel_size, stride=stride
        )
        clip_feature_image, _ = self.get_clip_feature_from_image(
            img, img_goal, kernel_size=kernel_size, stride=stride
        )
        combined_feature = clip_feature_text * 0.15 + clip_feature_image * 0.85
        # combined_feature = (combined_feature - combined_feature.min()) / (combined_feature.max() - combined_feature.min())
        return (
            clip_feature_text,
            clip_feature_image,
            np.clip(combined_feature, 0, 1),
            text_emb,
        )

    def get_clip_feature_c4_invariant(self, img, lan_goal, kernel_size=40, stride=20):
        with torch.no_grad():
            rgb = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0).to(self.device)

            patches = rgb.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(
                patches.shape[0] * patches.shape[1], -1, kernel_size, kernel_size
            )
            patches = patches.permute((1, 0, 2, 3))

            # group_list = [90, 180, 270]
            patches_rot_list = []
            for angle in range(3):
                patches_rot_list.append(torch.rot90(patches, angle, [2, 3]))

            patches_rot = torch.cat([patches, *patches_rot_list], axis=0)

            processor = clip_preprocess(224)
            patches_rot = processor(patches_rot / 255.0)

            text = self.tokenizer([lan_goal], padding=True, return_tensors="pt").to(
                self.device
            )
            text_features = self.tmodel(**text).text_embeds

            # patches = self.processor(
            #     images=patches,  # big patch image sent to CLIP
            #     return_tensors="pt",  # tell CLIP to return pytorch tensor
            # ).to(device).pixel_values  # too slow

            # score = self.model(**inputs)
            score = self.model(input_ids=text.input_ids, pixel_values=patches_rot)
            scores = score.logits_per_image
            scores = scores.reshape([4, -1, 1]).max(axis=0).values
            scores = scores.reshape(rgb.shape[0], patches.shape[0])
            # clip the scores
            scores = torch.clip(scores - scores.mean(dim=-1), 0, torch.inf)

            # normalize scores
            scores = (scores - scores.min(dim=-1).values) / (
                scores.max(dim=-1).values - scores.min(dim=-1).values
            )

            clip_feature = scores.reshape(
                rgb.shape[0],
                1,
                np.round((rgb.shape[2] - kernel_size - 1) / stride + 1).astype(int),
                np.round((rgb.shape[3] - kernel_size - 1) / stride + 1).astype(int),
            )
            clip_feature = nn.functional.interpolate(
                clip_feature,
                size=[rgb.shape[2], rgb.shape[3]],
                mode="bilinear",
                align_corners=True,
            )

            clip_feature = clip_feature.squeeze().unsqueeze(-1)
        return clip_feature.cpu().numpy(), text_features


def get_crop(rgb, pixel_xy, clip_kernel_size):
    pad_size = clip_kernel_size // 2
    pad_rgb = (
        F.pad(
            input=torch.from_numpy(rgb).permute(2, 0, 1),
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode="constant",
        )
        .permute(1, 2, 0)
        .numpy()
    )

    x, y = np.array(pixel_xy) + pad_size
    return pad_rgb[x - pad_size : x + pad_size, y - pad_size : y + pad_size, :]


def get_and_save_clip_features(
    data_folder_path, task_name, clip_feature_type="normal", use_image_goal=False
):
    """
    This function is only for simulation
    """
    # data_folder_path = "/home/mingxi/mingxi_ws/LEPP/cliport/data/"
    # task_name = "put-block-in-bowl-seen-colors-train"
    processor = CLIPFeatureProcessor(clip_feature_type)

    folder_path = os.path.join(data_folder_path, task_name)

    if clip_feature_type == "normal":
        clip_feature_type = ""
    clip_path = os.path.join(folder_path, "clip" + clip_feature_type)
    clip_pick_path = os.path.join(folder_path, "clip_pick" + clip_feature_type)
    clip_place_path = os.path.join(folder_path, "clip_place" + clip_feature_type)

    clip_crop_path = os.path.join(folder_path, "clip_crop" + clip_feature_type)
    crop_database_path = os.path.join(folder_path, "crop_database")
    clip_pick_crop_path = os.path.join(
        folder_path, "clip_pick_crop" + clip_feature_type
    )
    clip_place_crop_path = os.path.join(
        folder_path, "clip_place_crop" + clip_feature_type
    )

    clip_topdown_path = os.path.join(folder_path, "clip_topdown" + clip_feature_type)
    clip_topdown_crop_path = os.path.join(
        folder_path, "clip_topdown_crop" + clip_feature_type
    )

    color_path = os.path.join(folder_path, "color")
    lan_path = os.path.join(folder_path, "info")
    action_path = os.path.join(folder_path, "action")
    depth_path = os.path.join(folder_path, "depth")
    if not os.path.exists(clip_path):
        os.mkdir(clip_path)
    if not os.path.exists(clip_pick_path):
        os.mkdir(clip_pick_path)
    if not os.path.exists(clip_place_path):
        os.mkdir(clip_place_path)
    if not os.path.exists(clip_crop_path):
        os.mkdir(clip_crop_path)
    if not os.path.exists(clip_pick_crop_path):
        os.mkdir(clip_pick_crop_path)
    if not os.path.exists(clip_place_crop_path):
        os.mkdir(clip_place_crop_path)
    if not os.path.exists(clip_topdown_path):
        os.mkdir(clip_topdown_path)
    if not os.path.exists(clip_topdown_crop_path):
        os.mkdir(clip_topdown_crop_path)
    if not os.path.exists(crop_database_path):
        os.mkdir(crop_database_path)

    color_list = os.listdir(color_path)
    pbar = tqdm(total=len(color_list))
    crop_database_list = []
    pick_goal_list = []
    place_goal_list = []
    for color_name in color_list:
        episode_crop_database_list = []
        color = pickle.load(open(os.path.join(color_path, color_name), "rb"))
        info = pickle.load(open(os.path.join(lan_path, color_name), "rb"))
        action = pickle.load(open(os.path.join(action_path, color_name), "rb"))
        depth = pickle.load(open(os.path.join(depth_path, color_name), "rb"))
        features = []
        features_pick = []
        features_place = []
        features_crop = []
        features_pick_crop = []
        features_place_crop = []
        features_topdown_pp = []
        features_topdown_pick = []
        features_topdown_place = []
        features_topdown_pp_crop = []
        features_topdown_pick_crop = []
        features_topdown_place_crop = []
        for step in range(len(info)):
            lang_goal = info[step]["lang_goal"]
            pick_goal, place_goal = parse_instruction(task_name, lang_goal)
            print(lang_goal)
            print(f"pick_goal:{pick_goal}, place_goal:{place_goal}")

            image = color[step]
            dep = depth[step]
            rgb_views = []
            rgb_views_pick = []
            rgb_views_place = []
            rgb_views_crop = []
            rgb_views_pick_crop = []
            rgb_views_place_crop = []
            clip_kernel_size = 40

            bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
            pix_size = 0.003125
            cam_config = cameras.RealSenseD415.CONFIG
            obs = {"color": image, "depth": dep}
            cmap, _ = utils.get_fused_heightmap(obs, cam_config, bounds, pix_size)

            feat_combined_topdown_pick, _ = processor.get_clip_feature(
                cmap, pick_goal, clip_kernel_size
            )
            feat_combined_topdown_place, _ = processor.get_clip_feature(
                cmap, place_goal, clip_kernel_size
            )
            feat_combined_topdown_pp = (
                feat_combined_topdown_pick + feat_combined_topdown_place
            ) / 2
            # feat_combined_topdown_pp = (feat_combined_topdown_pp - feat_combined_topdown_pp.min()) / (feat_combined_topdown_pp.max()-feat_combined_topdown_pp.min())

            if action[step] is not None:
                p0_xyz = action[step]["pose0"][0]
                p1_xyz = action[step]["pose1"][0]
                p0 = utils.xyz_to_pix(p0_xyz, bounds, pix_size)
                p1 = utils.xyz_to_pix(p1_xyz, bounds, pix_size)

                crop0 = get_crop(cmap, p0, clip_kernel_size)
                crop1 = get_crop(cmap, p1, clip_kernel_size)
                (
                    _,
                    _,
                    feat_combined_topdown_pick_crop,
                    _,
                ) = processor.get_clip_feature_from_text_and_image(
                    cmap, pick_goal, crop0, clip_kernel_size
                )
                (
                    _,
                    _,
                    feat_combined_topdown_place_crop,
                    _,
                ) = processor.get_clip_feature_from_text_and_image(
                    cmap, place_goal, crop1, clip_kernel_size
                )
                # feat_combined_topdown_pp_crop = (feat_combined_topdown_pick_crop + feat_combined_topdown_place_crop)/2
                feat_combined_topdown_pp_crop = np.max(
                    np.concatenate(
                        [
                            feat_combined_topdown_pick_crop,
                            feat_combined_topdown_place_crop,
                        ],
                        axis=2,
                    ),
                    axis=2,
                )[..., None]
                # feat_combined_topdown_pp_crop = (feat_combined_topdown_pp_crop - feat_combined_topdown_pp_crop.min()) / (feat_combined_topdown_pp_crop.max()-feat_combined_topdown_pp_crop.min())
            else:
                feat_combined_topdown_pp_crop = feat_combined_topdown_pp
                feat_combined_topdown_pick_crop = feat_combined_topdown_pick
                feat_combined_topdown_place_crop = feat_combined_topdown_place

            for view_i in range(len(image)):
                # get feature for whole sentence goal
                # get feature for pick goal
                # get feature for place goal
                feat, _ = processor.get_clip_feature(
                    image[view_i], lang_goal, clip_kernel_size
                )
                rgb_views.append(feat)
                feat_pick, pick_text_emb = processor.get_clip_feature(
                    image[view_i], pick_goal, clip_kernel_size
                )
                rgb_views_pick.append(feat_pick)
                feat_place, place_text_emb = processor.get_clip_feature(
                    image[view_i], place_goal, clip_kernel_size
                )
                rgb_views_place.append(feat_place)

                pick_text_emb = processor.get_clip_text_feature(pick_goal)
                place_text_emb = processor.get_clip_text_feature(place_goal)

                if action[step] is not None:
                    (
                        _,
                        _,
                        feat_combined_pick,
                        _,
                    ) = processor.get_clip_feature_from_text_and_image(
                        image[view_i], pick_goal, crop0, clip_kernel_size
                    )
                    (
                        _,
                        _,
                        feat_combined_place,
                        _,
                    ) = processor.get_clip_feature_from_text_and_image(
                        image[view_i], place_goal, crop1, clip_kernel_size
                    )
                    feat_combined_pp = np.max(
                        np.concatenate(
                            [feat_combined_pick, feat_combined_place], axis=2
                        ),
                        axis=2,
                    )[..., None]
                    # feat_combined_pp = (feat_combined_pp - feat_combined_pp.min()) / (feat_combined_pp.max()-feat_combined_pp.min())
                    rgb_views_crop.append(feat_combined_pp)
                    rgb_views_pick_crop.append(feat_combined_pick)
                    rgb_views_place_crop.append(feat_combined_place)

                    episode_crop_database_list.append(
                        {
                            "pick_text_emb": pick_text_emb,
                            "pick_crop": crop0.astype(int),
                            "place_text_emb": place_text_emb,
                            "place_crop": crop1.astype(int),
                            "pick_obj_name": pick_goal,
                            "place_obj_name": place_goal,
                        }
                    )
                else:
                    # feat_combined_pp = np.zeros_like(image[view_i])[...,0:1]
                    rgb_views_crop.append(feat)
                    rgb_views_pick_crop.append(feat_pick)
                    rgb_views_place_crop.append(feat_place)

            pick_goal_list.append(pick_goal)
            place_goal_list.append(place_goal)

            features.append(np.stack(rgb_views))
            features_pick.append(np.stack(rgb_views_pick))
            features_place.append(np.stack(rgb_views_place))

            features_crop.append(np.stack(rgb_views_crop))
            features_pick_crop.append(np.stack(rgb_views_pick_crop))
            features_place_crop.append(np.stack(rgb_views_place_crop))

            features_topdown_pp.append(feat_combined_topdown_pp)
            features_topdown_pick.append(feat_combined_topdown_pick)
            features_topdown_place.append(feat_combined_topdown_place)

            features_topdown_pp_crop.append(feat_combined_topdown_pp_crop)
            features_topdown_pick_crop.append(feat_combined_topdown_pick_crop)
            features_topdown_place_crop.append(feat_combined_topdown_place_crop)

        features = (np.stack(features) * 255).astype(
            np.uint8
        )  # convert for storage efficiency
        features_pick = (np.stack(features_pick) * 255).astype(np.uint8)
        features_place = (np.stack(features_place) * 255).astype(np.uint8)
        features_crop = (np.stack(features_crop) * 255).astype(np.uint8)
        features_pick_crop = (np.stack(features_pick_crop) * 255).astype(np.uint8)
        features_place_crop = (np.stack(features_place_crop) * 255).astype(np.uint8)
        features_topdown = (
            np.concatenate(
                [
                    np.stack(features_topdown_pp),
                    np.stack(features_topdown_pick),
                    np.stack(features_topdown_place),
                ],
                axis=-1,
            )
            * 255
        ).astype(np.uint8)
        features_crop_topdown = (
            np.concatenate(
                [
                    np.stack(features_topdown_pp_crop),
                    np.stack(features_topdown_pick_crop),
                    np.stack(features_topdown_place_crop),
                ],
                axis=-1,
            )
            * 255
        ).astype(np.uint8)
        fname = color_name  # -{len(episode):06d}
        crop_database_list.append(episode_crop_database_list)
        with open(os.path.join(clip_path, fname), "wb") as f:
            pickle.dump(features, f)
        with open(os.path.join(clip_pick_path, fname), "wb") as f:
            pickle.dump(features_pick, f)
        with open(os.path.join(clip_place_path, fname), "wb") as f:
            pickle.dump(features_place, f)

        with open(os.path.join(clip_pick_crop_path, fname), "wb") as f:
            pickle.dump(features_pick_crop, f)
        with open(os.path.join(clip_place_crop_path, fname), "wb") as f:
            pickle.dump(features_place_crop, f)
        with open(os.path.join(clip_crop_path, fname), "wb") as f:
            pickle.dump(features_crop, f)

        with open(os.path.join(clip_topdown_path, fname), "wb") as f:
            pickle.dump(features_topdown, f)
        with open(os.path.join(clip_topdown_crop_path, fname), "wb") as f:
            pickle.dump(features_crop_topdown, f)

        pbar.update(1)

    with open(os.path.join(crop_database_path, "crop_database.npy"), "wb") as f:
        pickle.dump(crop_database_list, f)


# def main(cfg):
#     data_folder_path = "/home/mingxi/mingxi_ws/LEPP/cliport/data/"
#     task_name = "put-block-in-bowl-seen-colors-train"

#     folder_path = os.path.join(data_folder_path, task_name)
#     train_ds = RavensDataset(folder_path, cfg, n_demos=1000, augment=True)

#     processor = CLIP_processor()
#     clip_path = os.path.join(folder_path, 'clip')

#     for i in range(train_ds.n_episodes):
#         episode, _ = train_ds.load(i)
#         for traj in episode:
#             obs = traj[0]
#             color = obs['color']
#             lan_goal = traj[3]
#             feature = processor.get_clip_feature(color, lan_goal)

#             if not os.path.exists(clip_path):
#                 os.mkdir(clip_path)

#             fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
#             with open(os.path.join(clip_path, fname), 'wb') as f:
#                 pickle.dump(feature, f)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Example script to demonstrate argparse usage."
    )

    # Add arguments
    parser.add_argument(
        "data_path",
        type=str,
        default="/home/mingxi/mingxi_ws/LEPP/cliport/data/",
        help="An example of an optional argument",
    )
    parser.add_argument(
        "task", action="append", type=str, help="Add a value to the list"
    )
    parser.add_argument("mode", type=str, default="train", help="mode")
    parser.add_argument("cliptype", type=str, default="normal", help="cliptype")
    parser.add_argument("use_image_goal", type=bool, default=False, help="cliptype")

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    print("data_path:", args.data_path)
    print("task:", args.task)
    # data_folder_path = "/home/mingxi/mingxi_ws/LEPP/cliport/data/"
    # process_task_list = ["put-block-in-bowl-unseen-colors-val"]
    data_folder_path = args.data_path
    task_list = args.task
    mode = args.mode
    cliptype = args.cliptype
    use_image_goal = args.use_image_goal

    process_task_list = []
    for name in task_list:
        process_task_list.append(f"{name}-{mode}")

    for task in process_task_list:
        print(f"processing {task}")
        get_and_save_clip_features(
            data_folder_path, task, cliptype, use_image_goal=use_image_goal
        )

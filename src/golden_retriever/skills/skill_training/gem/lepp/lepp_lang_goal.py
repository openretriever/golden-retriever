import os
import time

import lepp.clip_revised as clip_revised
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from cliport.tasks import cameras
from cliport.utils import utils
from lepp.clip_preprocess import CLIP_processor
from lepp.clip_revised.clip import build_model
from lepp.dataset_tool import dataTool
from lepp.parser import parse_instruction
from lepp.pick_net import Attention
from lepp.place_net import Transport


class LEPPAgent(torch.nn.Module):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        # super().__init__(name, cfg, train_ds, test_ds)
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.total_steps = 0
        self.crop_size = 64

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg["train"]["val_repeats"]
        self.save_steps = cfg["train"]["save_steps"]

        self.use_image_goal = cfg["dataset"]["use_image_goal"]
        self.topdown = cfg["dataset"]["topdown"]
        self.image_text_ratio = cfg["dataset"]["image_text_ratio"]

        self.task = cfg["train"]["task"]

        if "multi" in self.task and ("multi-processed" not in self.task):
            if self.train_ds is not None:
                MULTI_TASKS_list = self.train_ds.MULTI_TASKS
            elif self.test_ds is not None:
                from cliport.dataset import MULTI_TASKS

                MULTI_TASKS_list = MULTI_TASKS
            else:
                print("no MULTI_TASKS list found")
        else:
            MULTI_TASKS_list = None

        if self.use_image_goal:
            if self.train_ds is not None:
                mode = "train"
                self.query_tool = dataTool(
                    self.train_ds, self.task, mode, MULTI_TASKS_list
                )
            elif self.test_ds is not None:
                mode = "val"
                self.query_tool = dataTool(
                    self.test_ds, self.task, mode, MULTI_TASKS_list
                )
            else:
                print("no crop base found")
        self.parse = cfg["lepp"]["enable_parse"]

        self.model_name = cfg["lepp"]["model_name"]
        self.pick_kernel_name = cfg["lepp"]["pick_kernel_name"]
        self.place_kernel_name = cfg["lepp"]["place_kernel_name"]
        self.vlm_name = cfg["lepp"]["vlm_name"]  # 'clip'
        self.init = True

        # ---------------------------------------------------------------------
        # clip_model_name = "RN50"
        clip_model_name = "ViT-B/32"
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device_type, "====")
        model, _ = clip_revised.load(clip_model_name, device=self.device_type)
        print("set up clip-ViT-B/32...")
        self.vlm_model = build_model(model.state_dict()).to(self.device_type)
        del model
        if clip_model_name == "RN50":
            self.lan_emb_dim = 1024
        else:
            self.lan_emb_dim = 512

        if cfg["dataset"]["type"] == "real":
            self.dist = "real"
        elif cfg["dataset"]["type"] == "realtable":
            self.dist = "realtable"
        elif cfg["dataset"]["type"] == "realspot":
            self.dist = "realspot"
        else:
            self.dist = "transporter"

        self._build_model()

        self.wb = cfg["wandb"]["enable"]
        if self.wb and (train_ds is not None):
            wandb.init(
                # set the wandb project where this run will be logged
                project="LEPP",
                name=cfg["wandb"]["name"],
                # track hyperparameters and run metadata
                config={
                    "agent_type": cfg["train"]["agent"],
                    "task_name": cfg["train"]["task"],
                    "n_demos": cfg["train"]["n_demos"],
                    "obs_net": self.model_name,
                },
            )

        if (train_ds is None) and (
            "real" not in self.dist
        ):  # if evaluation in real, use external clip
            self.clip_type = self.cfg["dataset"]["clip_type"]
            self.clip_processor = CLIP_processor(self.clip_type)

    def _build_model(self):
        self.attention = Attention(
            cfg=self.cfg,
            device=self.device_type,
            preprocess=utils.preprocess,
            init=self.init,
            model_name=self.model_name,
            kernel_name=self.pick_kernel_name,
            vlm_name=self.vlm_name,
            lan_kernel=True,
            dist=self.dist,
            vlm_model=self.vlm_model,
            lan_emb_dim=self.lan_emb_dim,
        )
        self.transport = Transport(
            cfg=self.cfg,
            device=self.device_type,
            preprocess=utils.preprocess,
            init=self.init,
            model_name=self.model_name,
            kernel_name=self.place_kernel_name,
            vlm_name=self.vlm_name,
            crop_size=self.crop_size,
            lan_kernel=True,
            dist=self.dist,
            vlm_model=self.vlm_model,
            lan_emb_dim=self.lan_emb_dim,
        )

    def train_new(self):
        # lepp implementation
        start_time = time.time()

        frame, _ = self.train_ds.get_sample()
        inp_img = frame["img"]
        inp_clip_features = frame["clip_features"]
        lan = frame["lang_goal"]
        p0 = frame["p0"]
        p0_theta = frame["p0_theta"]  # assume suction gripper
        p1, p1_theta = frame["p1"], frame["p1_theta"]

        subtask = self.train_ds.get_curr_task()

        # Get training losses.
        step = self.total_steps + 1
        loss0 = self.attention.train_step(
            inp_img, inp_clip_features, lan, p0, p0_theta, subtask
        )
        loss1 = self.transport.train_step(
            inp_img, inp_clip_features, lan, p0, p1, p1_theta, subtask
        )
        total_loss = loss0 + loss1
        self.total_steps = step
        losses = [loss0.item(), loss1.item()]

        if self.wb:
            wandb.log(
                {
                    "pick_loss": loss0.item(),
                    "place_loss": loss1.item(),
                    "loss": total_loss.item(),
                    "time_per_step": time.time() - start_time,
                }
            )

        return step, total_loss.item(), losses

    def act(
        self, obs, info, goal=None, parse_func=parse_instruction
    ):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        lang_goal = info["lang_goal"]
        subtask = self.test_ds.get_curr_task()
        pick_goal, place_goal = parse_func(self.task, lang_goal, subtask)
        rgb_views = []
        rgb_views_pick = []
        rgb_views_place = []
        image = obs["color"]
        img = self.test_ds.get_image(obs)
        rgb = img[..., :3]
        if self.use_image_goal:
            pick_emb = self.clip_processor.get_clip_text_feature(pick_goal)
            place_emb = self.clip_processor.get_clip_text_feature(place_goal)
            pick_crop = self.query_tool.query_crop(pick_emb)[0]
            place_crop = self.query_tool.query_crop(place_emb)[0]

        if self.topdown and self.use_image_goal:
            if pick_crop is not None:
                clip_feature_pick = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, pick_goal, pick_crop
                    )[2]
                )
            else:
                clip_feature_pick = self.clip_processor.get_clip_feature(
                    rgb, pick_goal
                )[0]
            if place_crop is not None:
                clip_feature_place = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, place_goal, place_crop
                    )[2]
                )
            else:
                clip_feature_place = self.clip_processor.get_clip_feature(
                    rgb, place_goal
                )[0]

            if not self.parse:
                clip_feature_pp = (clip_feature_pick + clip_feature_place) / 2
                clip_feature_pick = clip_feature_pp
                clip_feature_place = clip_feature_pp
        elif self.topdown and not self.use_image_goal:
            clip_feature_pp = self.clip_processor.get_clip_feature(rgb, lang_goal)[0]
            clip_feature_pick = self.clip_processor.get_clip_feature(rgb, pick_goal)[0]
            clip_feature_place = self.clip_processor.get_clip_feature(rgb, place_goal)[
                0
            ]

            if not self.parse:
                clip_feature_pick = clip_feature_pp
                clip_feature_place = clip_feature_pp
        elif not self.topdown and not self.use_image_goal:
            for view_i in range(len(image)):
                # get feature for whole sentence goal
                feat, _ = self.clip_processor.get_clip_feature(image[view_i], lang_goal)
                rgb_views.append(feat)
                # get feature for pick goal
                feat, _ = self.clip_processor.get_clip_feature(image[view_i], pick_goal)
                rgb_views_pick.append(feat)
                # get feature for place goal
                feat, _ = self.clip_processor.get_clip_feature(
                    image[view_i], place_goal
                )
                rgb_views_place.append(feat)
            rgb_views = np.stack(rgb_views)
            rgb_views_pick = np.stack(rgb_views_pick)
            rgb_views_place = np.stack(rgb_views_place)
            # Get heightmap from RGB-D images.
            if self.parse:
                clip_feature_pick = self.test_ds.get_clip_feature_image(
                    obs, rgb_views_pick
                )
                clip_feature_place = self.test_ds.get_clip_feature_image(
                    obs, rgb_views_place
                )
            else:
                clip_feature_pick = self.test_ds.get_clip_feature_image(obs, rgb_views)
                clip_feature_pick = clip_feature_place
        elif not self.topdown and self.use_image_goal:
            for view_i in range(len(image)):
                # get feature for whole sentence goal
                feat, _ = self.clip_processor.get_clip_feature(image[view_i], lang_goal)
                rgb_views.append(feat)
                # get feature for pick goal
                feat, _ = self.clip_processor.get_clip_feature(image[view_i], pick_goal)
                rgb_views_pick.append(feat)
                # get feature for place goal
                feat, _ = self.clip_processor.get_clip_feature(
                    image[view_i], place_goal
                )
                rgb_views_place.append(feat)
            rgb_views = np.stack(rgb_views)
            rgb_views_pick = np.stack(rgb_views_pick)
            rgb_views_place = np.stack(rgb_views_place)
            # Get heightmap from RGB-D images.

            # assert self.parse

            clip_feature_pick = self.test_ds.get_clip_feature_image(obs, rgb_views_pick)
            clip_feature_place = self.test_ds.get_clip_feature_image(
                obs, rgb_views_place
            )
            if pick_crop is not None:
                clip_feature_pick_topdown = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, pick_goal, pick_crop
                    )[2]
                )
                clip_feature_pick = (
                    clip_feature_pick * (1 - self.image_text_ratio)
                    + clip_feature_pick_topdown * self.image_text_ratio
                )

            if place_crop is not None:
                clip_feature_place_topdown = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, place_goal, place_crop
                    )[2]
                )
                clip_feature_place = (
                    clip_feature_place * (1 - self.image_text_ratio)
                    + clip_feature_place_topdown * self.image_text_ratio
                )

        self.attention.eval()
        self.transport.eval()
        with torch.no_grad():
            # Attention model forward pass.
            # pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
            clip_feature_pick = np.repeat(clip_feature_pick, 3, axis=-1)
            self.subtask = self.test_ds.get_curr_task()
            pick_conf = self.attention.forward(
                img, clip_feature_pick, lang_goal, self.subtask, softmax=True
            )
            # pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            # Transport model forward pass.
            place_inp = {"inp_img": img, "p0": p0_pix, "lang_goal": lang_goal}
            clip_feature_place = np.repeat(clip_feature_place, 3, axis=-1)
            place_conf = self.transport.forward(
                img, clip_feature_place, lang_goal, p0_pix, self.subtask, softmax=True
            )
            # place_conf = place_conf.squeeze(0)
            # place_conf = place_conf.permute(1, 2, 0)
            # place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            # Pixels to end effector poses.
            hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            "pose0": (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            "pose1": (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            "pick": [p0_pix[0], p0_pix[1], p0_theta],
            "place": [p1_pix[0], p1_pix[1], p1_theta],
        }

    def actReal(
        self, obs, info, goal=None, parse_func=parse_instruction
    ):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        clip_feature_pick = obs["clip_pick"]
        clip_feature_place = obs["clip_place"]
        img = obs["img"]

        lang_goal = info["lang_goal"]

        self.attention.eval()
        self.transport.eval()
        with torch.no_grad():
            # Attention model forward pass.
            # pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
            clip_feature_pick = np.repeat(clip_feature_pick, 3, axis=-1)
            pick_conf = self.attention.forward(
                img, clip_feature_pick, lang_goal, self.task, softmax=True
            )
            # pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            # Transport model forward pass.
            # place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
            clip_feature_place = np.repeat(clip_feature_place, 3, axis=-1)
            place_conf = self.transport.forward(
                img, clip_feature_place, lang_goal, p0_pix, self.task, softmax=True
            )
            # place_conf = place_conf.squeeze(0)
            # place_conf = place_conf.permute(1, 2, 0)
            # place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            # Pixels to end effector poses.
            hmap = img[:, :, 3]

        return {
            "pick": [p0_pix[0], p0_pix[1], p0_theta],
            "place": [p1_pix[0], p1_pix[1], p1_theta],
        }

    def actDualTable(self, obs1, obs2, info):
        """Run inference and return best action given visual observations."""
        clip_feature_pick1 = obs1["clip_pick"]
        clip_feature_place1 = obs1["clip_place"]
        img1 = obs1["img"]
        clip_feature_pick2 = obs2["clip_pick"]
        clip_feature_place2 = obs2["clip_place"]
        img2 = obs2["img"]

        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.margins(0,0)
        # vis = np.concatenate([clip_feature_place2, clip_feature_pick2, np.zeros_like(clip_feature_pick2)], axis=-1)
        # plt.imshow(vis)
        # plt.savefig("a.png",bbox_inches='tight')

        lang_goal = info["lang_goal"]

        self.attention.eval()
        self.transport.eval()
        with torch.no_grad():
            # Attention model forward pass.
            clip_feature_pick1 = np.repeat(clip_feature_pick1, 3, axis=-1)
            clip_feature_pick2 = np.repeat(clip_feature_pick2, 3, axis=-1)
            pick_conf1 = self.attention.forward(
                img1, clip_feature_pick1, lang_goal, self.task, softmax=False
            )
            pick_conf2 = self.attention.forward(
                img2, clip_feature_pick2, lang_goal, self.task, softmax=False
            )
            pick_conf = torch.concat([pick_conf1, pick_conf2], axis=-1)
            # pick_conf = pick_conf.detach().cpu().numpy()
            pick_conf_shape = pick_conf.shape
            pick_conf = pick_conf.reshape(-1)
            pick_conf = F.softmax(pick_conf, dim=-1)
            pick_conf = pick_conf.reshape(pick_conf_shape[1:]).detach().cpu().numpy()
            pick_conf = pick_conf.transpose(1, 2, 0)
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            if p0_pix[1] < pick_conf1.shape[-1]:
                pick_table = 1
                image_selected = img1
                p0_selected = p0_pix
            else:
                pick_table = 2
                image_selected = img2
                p0_selected = p0_pix - np.array([0, pick_conf1.shape[-1]])

            # Transport model forward pass.
            # place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
            clip_feature_place1 = np.repeat(clip_feature_place1, 3, axis=-1)
            clip_feature_place2 = np.repeat(clip_feature_place2, 3, axis=-1)
            place_conf1 = self.transport.forward(
                img1,
                clip_feature_place1,
                lang_goal,
                p0_selected,
                self.task,
                softmax=False,
                crop_source=image_selected,
            )
            place_conf2 = self.transport.forward(
                img2,
                clip_feature_place2,
                lang_goal,
                p0_selected,
                self.task,
                softmax=False,
                crop_source=image_selected,
            )
            place_conf = torch.concat([place_conf1, place_conf2], axis=-1)
            place_conf_shape = place_conf.shape
            place_conf = place_conf.reshape(-1)
            place_conf = F.softmax(place_conf, dim=-1)
            place_conf = place_conf.reshape(place_conf_shape[1:]).detach().cpu().numpy()
            place_conf = place_conf.transpose(1, 2, 0)

            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            if p1_pix[1] < place_conf1.shape[-1]:
                place_table = 1
                p1_selected = p1_pix
            else:
                place_table = 2
                p1_selected = p1_pix - np.array([0, place_conf1.shape[-1]])

        return {
            "pick_table": pick_table,
            "pick": [p0_selected[0], p0_selected[1], p0_theta],
            "place_table": place_table,
            "place": [p1_selected[0], p1_selected[1], p1_theta],
        }

    def save(self, models_dir):
        # if not os.path.exists(models_dir):
        #     os.makedirs(models_dir)
        # attention_fname1 = 'attention-ckpt-steps=%d.pt' % self.total_steps
        # attention_fname2 = 'transport-ckpt-steps=%d.pt' % self.total_steps
        # attention_fname1 = os.path.join(f"{models_dir}/checkpoints/", attention_fname1)
        # attention_fname2 = os.path.join(f"{models_dir}/checkpoints/", attention_fname2)
        # torch.save(self.attention, attention_fname1)
        # torch.save(self.transport, attention_fname2)
        print(models_dir)
        checkpoint_path = f"{models_dir}/checkpoints/"
        model_path = "steps=%d.pt" % self.total_steps
        model_path = os.path.join(checkpoint_path, model_path)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.eval()
        torch.save(self.state_dict(), model_path)
        self.train()
        print(f"Save {models_dir} model at {self.total_steps} iterations.")

    def load(self, model_path):
        # self.load_state_dict(torch.load(model_path)['state_dict'])
        self.load_state_dict(torch.load(model_path))
        self.to(device=self.device_type)

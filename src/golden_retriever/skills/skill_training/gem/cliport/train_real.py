"""Main training script."""

import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch.backends.cudnn as cudnn
from cliport import agents
from cliport.dataset_real import RealDataset

cudnn.benchmark = True


def train(self, dataset, lan_key=None, show_lan=False):
    time_0 = time.time()
    img, p0, p0_theta, lan, checker = self.get_sample(
        dataset, augment=True, lan_key=lan_key
    )
    step = self.total_steps + 1
    if show_lan:
        print(lan)
    loss = self.attention.train(img, lan, p0, p0_theta)
    time_0 = time.time() - time_0
    self.local_log.append(loss)
    print(f"Train Iter: {step} Loss: {loss:.4f} time: {time_0:.4f}")
    self.total_steps = step


@hydra.main(config_path="./cfg", config_name="train")
def main(cfg):
    # Logger
    np.random.seed(
        cfg["lepp"]["seed"]
    )  # make sure the training data is same among methods
    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg["train"]["train_dir"], "checkpoints")
    last_checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
    last_checkpoint = (
        last_checkpoint_path
        if os.path.exists(last_checkpoint_path) and cfg["train"]["load_from_last_ckpt"]
        else None
    )

    # Trainer
    max_epochs = cfg["train"]["n_steps"] // cfg["train"]["n_demos"]

    # Config
    data_dir = cfg["train"]["data_dir"]
    task = cfg["train"]["task"]
    agent_type = cfg["train"]["agent"]
    n_demos = cfg["train"]["n_demos"]
    n_val = cfg["train"]["n_val"]
    name = "{}-{}-{}".format(task, agent_type, n_demos)
    n_steps = cfg["train"]["n_steps"]

    save_freq_step = cfg["train"]["save_freq_step"]
    val_freq_step = save_freq_step

    # Datasets
    dataset_type = cfg["dataset"]["type"]
    if "multi" in dataset_type:
        train_ds = RealDataset(
            data_dir, cfg, group=task, mode="train", n_demos=n_demos, augment=True
        )
        # val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RealDataset(
            os.path.join(data_dir, "{}-train".format(task)),
            cfg,
            n_demos=n_demos,
            augment=True,
        )
        # val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_ds, None)

    log = {}

    # train agent and save snapshot
    pick_loss_list = []
    place_loss_list = []
    loss_list = []
    while agent.total_steps < n_steps:
        time0 = time.time()
        step, total_loss, losses = agent.train_new()

        pick_loss_list.append(losses[0])
        pick_loss_list.append(losses[1])
        loss_list.append(total_loss)
        print(
            f"Train Iter: {step} PickLoss: {losses[0]:.7f} PlaceLoss: {losses[1]:.7f} Loss: {total_loss:.7f} Step time: {time.time()-time0}"
        )

        if (step != 0) and (step % save_freq_step) == 0:
            agent.save(models_dir=cfg["train"]["log_dir"])
            log["pick_loss"] = pick_loss_list
            log["place_loss"] = place_loss_list
            log["total_loss"] = loss_list
            np.save(os.path.join(cfg["train"]["log_dir"], "log.npy"), log)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# System
import argparse
import random
import time
import pathlib as pl
import typing as ty

# Third Party
import torch as t
import tqdm
import numpy as np
import itk
import roma as rm

# Internal
import dv


class AAANetRunner:
    """
    A class for AAANet training and inference.
    """

    def __init__(self, exp_path: pl.Path):
        """Initialize the AAANetRunner class.

        Args:
            exp_path: Path to the `toml` format experiment file.
        """
        self.exp_path = exp_path
        self.exp = dv.utils.Experiment(self.exp_path)

        self.set_seeds()

        self.model = dv.models.AAANet(self.exp).to(self.exp.device)
        self.model.coarse_requires_grad(not self.exp.coarse_frozen)
        self.model.fine_requires_grad(not self.exp.fine_frozen)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.exp.lr_initial)
        self.scaler = t.cuda.amp.GradScaler()

        if self.exp.loss_function == "cross_entropy":
            self.loss_fn = t.nn.CrossEntropyLoss(
                weight=t.Tensor(self.exp.class_weights).to(device=self.exp.device)
            )
        elif self.exp.loss_function == "jaccard":
            self.loss_fn = dv.loss_functions.JaccardLoss()
        else:
            assert False, f"Loss function '{self.exp.loss_function}' not recognized."

        self.dataset = dv.datasets.BasicDataset(self.exp)

        # Training
        self.dataset_trn = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_trn_indices(),
        )
        self.dataloader_trn = t.utils.data.DataLoader(
            self.dataset_trn,
            batch_size=self.exp.batch_size,
            shuffle=True,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )
        self.loss_seg_coarse_trn = []
        self.loss_centroid_trn = []
        self.loss_transform_trn = []
        self.loss_seg_fine_trn = []
        self.time_trn = []

        # Validation
        self.dataset_val = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_val_indices(),
        )
        self.dataloader_val = t.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.exp.batch_size,
            shuffle=False,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )
        self.loss_seg_coarse_val = []
        self.loss_centroid_val = []
        self.loss_transform_val = []
        self.loss_seg_fine_val = []
        self.time_val = []

        # Testing
        self.dataset_tst = t.utils.data.Subset(
            self.dataset,
            self.dataset.get_tst_indices(),
        )
        self.dataloader_tst = t.utils.data.DataLoader(
            self.dataset_tst,
            batch_size=self.exp.batch_size,
            shuffle=False,
            num_workers=self.exp.num_workers,
            pin_memory=self.exp.pin_memory,
        )
        self.loss_seg_coarse_tst = []
        self.loss_centroid_tst = []
        self.loss_transform_tst = []
        self.loss_seg_fine_tst = []
        self.time_tst = []

    def set_seeds(self):
        """Set PyTorch, Numpy, and Python seeds to reduce stochastic effects."""
        t.manual_seed(self.exp.seed)
        random.seed(self.exp.seed)
        np.random.seed(self.exp.seed)

    def train(self):
        """Alternate between training and validation runs for the number of epochs specified in the experiment file."""
        self.exp.dir_output.mkdir(parents=True, exist_ok=True)
        for e in range(self.exp.epochs):
            self.epoch = e
            self.run_epoch("trn")
            self.run_epoch("val")
            self.save_checkpoint()

            print(f"Coarse loss: {np.mean(self.loss_seg_coarse_val[-1]):.4}")
            print(f"Centroid loss: {np.mean(self.loss_centroid_val[-1]):.4}")
            print(f"Transform loss: {np.mean(self.loss_transform_val[-1]):.4}")
            print(f"Fine loss: {np.mean(self.loss_seg_fine_val[-1]):.4}")

    def test(self):
        """Run on the test set."""
        self.run_epoch("tst", show=True)
        self.save_checkpoint()

        print(f"Coarse loss: {np.mean(self.loss_seg_coarse_tst[-1]):.4}")
        print(f"Centroid loss: {np.mean(self.loss_centroid_tst[-1]):.4}")
        print(f"Transform loss: {np.mean(self.loss_transform_tst[-1]):.4}")
        print(f"Fine loss: {np.mean(self.loss_seg_fine_tst[-1]):.4}")

    def run_epoch(self, mode: ty.Literal["trn", "val", "tst"], show: bool = False):
        """Run a single epoch.

        Args:
            mode: Specify training, validation, or testing.
            show: Specify whether to display the result.
        """

        assert mode in {"trn", "val", "tst"}, "Mode not recognized."

        if mode == "trn":
            self.model.train()
            loop = tqdm.tqdm(self.dataloader_trn)
            self.dataset.augmentation = self.exp.augmentation
        elif mode == "val":
            self.model.eval()
            loop = tqdm.tqdm(self.dataloader_val)
            self.dataset.augmentation = False
        elif mode == "tst":
            self.model.eval()
            loop = tqdm.tqdm(self.dataloader_tst)
            self.dataset.augmentation = False
        else:
            assert False, "Mode not recognized."

        N = len(loop)

        loop.set_postfix(epoch=self.epoch)
        epoch_seg_coarse_loss = []
        epoch_centroid_loss = []
        epoch_transform_loss = []
        epoch_seg_fine_loss = []

        start_time = time.perf_counter()

        for i, gt in enumerate(loop):
            gt["img"] = gt["img"].to(device=self.exp.device)
            gt["seg"] = gt["seg"].to(device=self.exp.device)
            gt["seg_coarse"] = t.nn.functional.grid_sample(
                gt["seg"],
                self.model.coarse_points,
                mode="bilinear",
                align_corners=False,
            )

            transform = dv.geometry.transform_from_seg(
                gt["seg_coarse"][:, 1:, :].sum(1),
                self.exp.get_new_axis().to(self.exp.device),
            )

            gt["transform"] = rm.RigidUnitQuat(
                linear=transform["quaternion"], translation=transform["centroid"]
            )

            # Forward
            with t.inference_mode(mode in {"val", "tst"}):
                with t.cuda.amp.autocast():
                    pd = self.model(gt["img"])

                    # Upsample segmentation to original resolution.
                    pd["coarse_upsampled"] = t.nn.functional.grid_sample(
                        pd["coarse"],
                        self.exp.input_points,
                        mode="bilinear",
                        align_corners=False,
                        padding_mode="border",
                    )
                    if show:
                        dv.visualization.Viewer3D(
                            gt["img"][0, 0].cpu(),
                            t.argmax(pd["coarse_upsampled"], dim=1)[0].cpu(),
                            self.exp.labels,
                            slices=[0, 1, 2],
                        )
                    seg_coarse_loss = self.loss_fn(pd["coarse_upsampled"], gt["seg"])

                    pd["transform_i"] = pd["transform"].normalize().inverse()

                    fp_i = self.model.calculate_fine_points(
                        pd["transform_i"], self.exp.input_shape(), inverse=True
                    ).to(self.exp.device)

                    pd["fine_upsampled"] = t.nn.functional.grid_sample(
                        pd["fine"],
                        fp_i,
                        mode="bilinear",
                        align_corners=False,
                        padding_mode="border",
                    )
                    if show:
                        fp = self.model.calculate_fine_points(
                            pd["transform"], self.exp.fine_shape
                        ).to(self.exp.device)
                        img_fine = t.nn.functional.grid_sample(
                            gt["img"], fp, mode="bilinear", align_corners=False
                        )
                        dv.visualization.Viewer3D(
                            img_fine[0, 0].cpu(),
                            t.argmax(pd["fine"], dim=1)[0].cpu(),
                            self.exp.labels,
                            slices=[0, 1, 2],
                        )

                    seg_fine_loss = self.loss_fn(pd["fine_upsampled"], gt["seg"])
                    seg_fine_loss_weight = self.exp.fine_loss_weight(self.epoch)

                    sum_loss = (
                        seg_coarse_loss * (1 - seg_fine_loss_weight)
                        + seg_fine_loss * seg_fine_loss_weight
                    )

                    centroid_error = (
                        gt["transform"].translation - pd["transform"].translation
                    )
                    centroid_error = centroid_error.flip(1)
                    centroid_error = centroid_error * self.exp.coarse_fov / 2
                    centroid_error = centroid_error.pow(2).sum(dim=1).pow(0.5)

                    angle_error = dv.geometry.angle(
                        pd["axis"], transform["axis"]
                    ).squeeze(-1)

                    epoch_seg_coarse_loss += seg_coarse_loss.tolist()
                    epoch_centroid_loss += centroid_error.tolist()
                    epoch_transform_loss += angle_error.tolist()
                    epoch_seg_fine_loss += seg_fine_loss.tolist()

            # Backward
            if mode == "trn":
                sum_loss_scale = dv.utils.uneven_batch_size(
                    i, self.exp.batch_size_virtual, N
                )
                self.scaler.scale(sum_loss / sum_loss_scale).backward()
                if ((i + 1) % self.exp.batch_size_virtual == 0) or ((i + 1) == N):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

        epoch_time = time.perf_counter() - start_time

        if mode == "trn":
            self.loss_seg_coarse_trn.append(epoch_seg_coarse_loss)
            self.loss_centroid_trn.append(epoch_centroid_loss)
            self.loss_transform_trn.append(epoch_transform_loss)
            self.loss_seg_fine_trn.append(epoch_seg_fine_loss)
            self.time_trn.append(epoch_time)
        elif mode == "val":
            self.loss_seg_coarse_val.append(epoch_seg_coarse_loss)
            self.loss_centroid_val.append(epoch_centroid_loss)
            self.loss_transform_val.append(epoch_transform_loss)
            self.loss_seg_fine_val.append(epoch_seg_fine_loss)
            self.time_val.append(epoch_time)
        elif mode == "tst":
            self.loss_seg_coarse_tst.append(epoch_seg_coarse_loss)
            self.loss_centroid_tst.append(epoch_centroid_loss)
            self.loss_transform_tst.append(epoch_transform_loss)
            self.loss_seg_fine_tst.append(epoch_seg_fine_loss)
            self.time_tst.append(epoch_time)
        else:
            assert False, "Mode not recognized."

    def save_checkpoint(self):
        """Save `checkpoint_last` (always).  Additionally save `checkpoint_best` if this run represents the lowest recorded validation error (or if none has been previously recorded)."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_seg_coarse_trn": self.loss_seg_coarse_trn,
            "loss_centroid_trn": self.loss_centroid_trn,
            "loss_transform_trn": self.loss_transform_trn,
            "loss_seg_fine_trn": self.loss_seg_fine_trn,
            "time_trn": self.time_trn,
            "loss_seg_coarse_val": self.loss_seg_coarse_val,
            "loss_centroid_val": self.loss_centroid_val,
            "loss_transform_val": self.loss_transform_val,
            "loss_seg_fine_val": self.loss_seg_fine_val,
            "time_val": self.time_val,
            "loss_seg_coarse_tst": self.loss_seg_coarse_tst,
            "loss_centroid_tst": self.loss_centroid_tst,
            "loss_transform_tst": self.loss_transform_tst,
            "loss_seg_fine_tst": self.loss_seg_fine_tst,
            "time_tst": self.time_tst,
        }
        t.save(checkpoint, self.exp.get_checkpoint_last_path())
        if len(self.loss_seg_fine_val) <= 1 or self.loss_seg_fine_val[-1] < min(
            self.loss_seg_fine_val[0:-1]
        ):
            print(f"New lowest validation loss; saving checkpoint.")
            t.save(checkpoint, self.exp.get_checkpoint_best_path())

    def load_checkpoint(self, path: pl.Path):
        """Load checkpoint from path.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = t.load(path)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.loss_seg_coarse_trn = checkpoint["loss_seg_coarse_trn"]
        self.loss_centroid_trn = checkpoint["loss_centroid_trn"]
        self.loss_transform_trn = checkpoint["loss_transform_trn"]
        self.loss_seg_fine_trn = checkpoint["loss_seg_fine_trn"]
        self.time_trn = checkpoint["time_trn"]

        self.loss_seg_coarse_val = checkpoint["loss_seg_coarse_val"]
        self.loss_centroid_val = checkpoint["loss_centroid_val"]
        self.loss_transform_val = checkpoint["loss_transform_val"]
        self.loss_seg_fine_val = checkpoint["loss_seg_fine_val"]
        self.time_val = checkpoint["time_val"]

        self.loss_seg_coarse_tst = checkpoint["loss_seg_coarse_tst"]
        self.loss_centroid_tst = checkpoint["loss_centroid_tst"]
        self.loss_transform_tst = checkpoint["loss_transform_tst"]
        self.loss_seg_fine_tst = checkpoint["loss_seg_fine_tst"]
        self.time_tst = checkpoint["time_tst"]

    def load_checkpoint_helper(self, option: ty.Literal["pretrain", "last", "best"]):
        """Load a specific checkpoint.

        Args:
            option: Specify whether to load the pretrain, last, or best checkpoint.
        """
        if option == "pretrain":
            self.load_checkpoint(self.exp.get_checkpoint_pretrain_path())
        elif option == "last":
            self.load_checkpoint(self.exp.get_checkpoint_last_path())
        elif option == "best":
            self.load_checkpoint(self.exp.get_checkpoint_best_path())
        else:
            assert False, "Option not recognized."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=pl.Path)
    parser.add_argument("--load-checkpoint", choices=("pretrain", "last", "best"))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    runner = AAANetRunner(args.experiment)

    if args.load_checkpoint is not None:
        runner.load_checkpoint_helper(args.load_checkpoint)
    if args.train:
        runner.train()
    if args.test:
        runner.test()


if __name__ == "__main__":
    main()

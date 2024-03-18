# System
import pathlib as pl

# Third Party
import itk
import tomlkit as tk
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch as t

# Internal
from . import geometry


class Label:
    """Segmentation label, to be used with Viewer3D."""

    def __init__(
        self, integer: int, color: list[float], label: str, legend: bool = True
    ):
        self.integer = integer
        self.color = color
        self.label = label
        self.legend = legend


def tex(macro: str, value):
    """Create a LaTeX macro.

    Args:
        macro: Name of the LaTeX macro.
        value: Value of the LaTeX macro.
    """
    return f"\\newcommand{{\\{macro}}}{{{value}}}"


def print_itk_info(image):
    """Print basic information about an ITK image."""
    size = np.asarray(itk.size(image))
    origin = np.asarray(itk.origin(image))
    spacing = np.asarray(itk.spacing(image))
    direction = np.asarray(image.GetDirection())
    fov = np.asarray([sz * sp for sz, sp in zip(size, spacing)])

    print(f"Size: {size}")
    print(f"Origin: {origin}")
    print(f"Spacing: {np.diag(spacing)}")
    print(f"Direction: {direction}")
    print(f"FOV: {fov}")


class Experiment:
    """Experiment file, instantiated from a TOML file path."""

    def __init__(self, config: pl.Path):
        """Initialization.

        Args:
            config: Path to the TOML configuration file.
        """
        with open(config, mode="rt", encoding="utf-8") as fp:
            exp = tk.load(fp)
        self.dir_trn = pl.Path(exp["dir_trn"])
        self.dir_val = pl.Path(exp["dir_val"])
        self.dir_tst = pl.Path(exp["dir_tst"])
        self.dir_output = pl.Path(exp["dir_output"])
        self.network_title = exp["network_title"]
        self.img_file = exp["img_file"]
        self.seg_file = exp["seg_file"]
        self.checkpoint_pretrain_file_name = exp["checkpoint_pretrain_file_name"]
        self.checkpoint_last_file_name = exp["checkpoint_last_file_name"]
        self.checkpoint_best_file_name = exp["checkpoint_best_file_name"]
        self.loss_file_name = exp["loss_file_name"]

        self.itk_num_threads = exp["itk_num_threads"]
        self.device = exp["device"]
        self.num_workers = exp["num_workers"]
        self.pin_memory = exp["pin_memory"]

        self.seed = exp["seed"]
        self.epochs = exp["epochs"]

        self.coarse_spacing = exp["coarse_spacing"]
        self.coarse_shape = exp["coarse_shape"]
        self.coarse_features = exp["coarse_features"]
        self.coarse_frozen = exp["coarse_frozen"]
        self.coarse_fov = t.Tensor(self.coarse_shape) * self.coarse_spacing
        self.coarse_fov = self.coarse_fov.to(self.device)

        self.fine_spacing = exp["fine_spacing"]
        self.fine_shape = exp["fine_shape"]
        self.fine_features = exp["fine_features"]
        self.fine_frozen = exp["fine_frozen"]
        self.fine_fov = t.Tensor(self.fine_shape) * self.fine_spacing
        self.fine_fov = self.fine_fov.to(self.device)

        self.channels = exp["channels"]
        self.classes = exp["classes"]

        self.batch_size = exp["batch_size"]
        self.batch_size_virtual = exp["batch_size_virtual"]

        self.air_hu = exp["air_hu"]
        self.clip_min = exp["clip_min"]
        self.clip_max = exp["clip_max"]
        self.label_map = {k: v for k, v in exp["label_map"]}

        self.augmentation = exp["augmentation"]

        self.normalization = exp["normalization"]
        self.pass_features = exp["pass_features"]
        self.orthogonal = exp["orthogonal"]

        self.lr_initial = exp["lr_initial"]

        self.loss_function = exp["loss_function"]
        self.loss_weight_strategy = exp["loss_weight_strategy"]
        self.loss_weight_fine = exp["loss_weight_fine"]
        self.loss_weight_slope = exp["loss_weight_slope"]
        self.loss_weight_hinge = exp["loss_weight_hinge"]
        self.loss_weight_bleed = exp["loss_weight_bleed"]

        assert len(self.coarse_shape) == len(self.fine_shape)
        assert self.dim() in {2, 3}

        self.set_itk_num_threads()

        self.labels = [
            Label(0, (0, 0, 0, 0), "Background", legend=False),
            Label(1, (1, 0, 0, 0.5), "Aorta"),
        ]

        self.input_points = self.calculate_input_points()

    def calculate_input_points(self):
        """Calculate the points of the full FOV, high-resolution input image."""
        I = geometry.affine_identity(self.dim(), batch_size=self.batch_size)
        points = t.nn.functional.affine_grid(
            I,
            [self.batch_size, self.channels, *self.input_shape()],
            align_corners=False,
        ).half()
        points = points.to(self.device)
        return points

    def __repr__(self):
        return f"{self.__dict__}"

    def __str__(self):
        items = [f"{k} : {v}" for k, v in self.__dict__.items()]
        return "\n".join(items)

    def set_itk_num_threads(self):
        """Set the global number of ITK threads."""
        threader = itk.MultiThreaderBase.New()
        threader.SetGlobalDefaultNumberOfThreads(self.itk_num_threads)

    def input_shape(self):
        """Calculate the input shape."""
        return (
            np.array(self.coarse_shape) * self.coarse_spacing / self.fine_spacing
        ).astype(int)

    def input_shape_itk(self):
        """Calculate the input shape, reversed to match ITK convention."""
        return np.array(list(reversed(self.input_shape())))

    def dim(self):
        """Return the dimension of the input image."""
        return len(self.coarse_shape)

    def get_checkpoint_pretrain_path(self):
        """Get the path to the pretrained model."""
        return self.dir_output / self.checkpoint_pretrain_file_name

    def get_checkpoint_last_path(self):
        """Get the path to the last saved model."""
        return self.dir_output / self.checkpoint_last_file_name

    def get_checkpoint_best_path(self):
        """Get the path to the best saved model."""
        return self.dir_output / self.checkpoint_best_file_name

    def get_new_axis(self):
        """Get the new axis to which the points will be aligned."""
        return t.Tensor([[0, 0, 1]]).repeat(self.batch_size, 1).half()

    def get_identity_quat(self):
        """Get the identity quaternion."""
        return t.Tensor([[0, 0, 0, 1]]).repeat(self.batch_size, 1).half()

    def fine_channels(self):
        """Calculate the number of channels, depending on whether feature sharing is enabled."""
        if self.pass_features:
            return self.channels + self.classes
        else:
            return self.channels

    def fine_loss_weight(self, epoch: int):
        """Calculate the loss weight as a function of epoch and loss strategy.

        Args:
            epoch: Current epoch.
        """
        if self.loss_weight_strategy == "constant":
            return self.loss_weight_fine
        elif self.loss_weight_strategy == "linear":
            return (
                epoch / self.epochs * (1 - self.loss_weight_bleed)
                + self.loss_weight_bleed / 2
            )
        elif self.loss_weight_strategy == "sigmoid":
            s = self.loss_weight_slope
            h = self.loss_weight_hinge
            b = self.loss_weight_bleed
            return logistic(epoch, s, h, b)
        else:
            assert (
                False
            ), f"Provided loss strategy {self.loss_weight_strategy} not recognized."


def logistic(x: float, slope: float, hinge: float, bleed: float):
    """Logistic function."""
    return 1 / (1 + np.e ** (-slope * (x - hinge))) * (1 - bleed) + bleed / 2


def uneven_batch_size(index: int, batch_size: int, n_samples: int):
    """Calculate the batch size for a given index, given that the batch does not necessarily divide evenly into the total number of samples.

    Args:
        index: Current index.
        batch_size: Batch size.
        n_samples: Total number of samples.
    """
    if index < (n_samples // batch_size) * batch_size:
        return batch_size
    else:
        return n_samples % batch_size

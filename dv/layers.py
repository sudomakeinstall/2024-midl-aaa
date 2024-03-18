# System
import typing as ty

# Third Party
import torch as t
import numpy as np

conv_dict = {1: t.nn.Conv1d, 2: t.nn.Conv2d, 3: t.nn.Conv3d}
conv_t_dict = {
    1: t.nn.ConvTranspose1d,
    2: t.nn.ConvTranspose2d,
    3: t.nn.ConvTranspose3d,
}
batch_norm_dict = {1: t.nn.BatchNorm1d, 2: t.nn.BatchNorm2d, 3: t.nn.BatchNorm3d}
max_pool_dict = {1: t.nn.MaxPool1d, 2: t.nn.MaxPool2d, 3: t.nn.MaxPool3d}


class CvNmAc(t.nn.Module):
    """A layer containing one or more convolutions, a normalization, and an activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: ty.Literal["group", "instance", "layer", "batch"] = "group",
        channels_per_group: int = 8,
        convolutions: int = 2,
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding.
            norm: Normalization strategy.
            channels_per_group: For use with Group Normalization.
            convolutions: Number of convolutions.
        """
        super(CvNmAc, self).__init__()
        modules = t.nn.ModuleList()

        # Convolution
        for c in range(convolutions):
            modules.append(
                conv_dict[dimension](
                    in_channels if c == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )

        # Normalization
        if norm == "group":
            modules.append(
                t.nn.GroupNorm(out_channels // channels_per_group, out_channels)
            )
        elif norm == "instance":
            modules.append(t.nn.GroupNorm(out_channels, out_channels))
        elif norm == "layer":
            modules.append(t.nn.GroupNorm(1, out_channels))
        elif norm == "batch":
            modules.append(batch_norm_dict[dimension](out_channels))
        else:
            assert (
                False
            ), f"Option norm='{norm}' not recognized! Allowed options are 'group', 'instance', 'layer', or 'batch'."

        # Activation
        modules.append(t.nn.ReLU(inplace=True))

        self.conv = t.nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class CvNmAcxN(t.nn.Module):
    """A layer which repeats `CvNmAc` a specified number of times."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: ty.Literal["group", "instance", "layer", "batch"] = "group",
        channels_per_group: int = 8,
        convolutions: int = 2,
        repetitions: int = 2,
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding.
            norm: Normalization strategy.
            channels_per_group: For use with Group Normalization.
            convolutions: Number of convolutions.
            repetitions: Number of times to repeat `CvNmAc`.
        """
        super(CvNmAcxN, self).__init__()
        modules = t.nn.ModuleList()

        for r in range(repetitions):
            modules.append(
                CvNmAc(
                    in_channels if r == 0 else out_channels,
                    out_channels,
                    dimension,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    norm=norm,
                    channels_per_group=channels_per_group,
                    convolutions=convolutions,
                )
            )

        self.conv = t.nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

# System
import typing as ty

# Third Party
import torch as t
import numpy as np
import roma as rm

# Internal
from . import layers
from . import geometry
from . import utils


class AAANet(t.nn.Module):
    """A neural network containing coarse segmentation, transformation, and fine segmentation modules."""

    def __init__(self, exp: utils.Experiment):
        """Initialization.

        Args:
            exp: Experiment instance.
        """
        super(AAANet, self).__init__()
        self.exp = exp

        self.unet_coarse = UNet(
            self.exp.channels,
            self.exp.classes,
            self.exp.dim(),
            self.exp.coarse_features,
            self.exp.normalization,
        )
        # TODO
        # fc_len = self.calculate_fully_connected_length()
        # self.fc = t.nn.Linear(fc_len, self.exp.dim() + 4)
        self.unet_fine = UNet(
            self.exp.fine_channels(),
            self.exp.classes,
            self.exp.dim(),
            self.exp.fine_features,
            self.exp.normalization,
        )
        self.coarse_points = self.get_affine_grid(self.exp.coarse_shape)

    def forward(self, x):
        img_coarse = t.nn.functional.grid_sample(
            x, self.coarse_points, mode="bilinear", align_corners=False
        )

        coarse = self.unet_coarse(img_coarse)

        # TODO
        # transform = self.fc(t.flatten(hook, start_dim=1))
        transform = geometry.transform_from_seg(
            coarse["seg"].softmax(1)[:, 1:, :].sum(1),
            self.exp.get_new_axis().to(self.exp.device),
        )

        if self.exp.orthogonal:
            transform["quaternion"] = self.exp.get_identity_quat().to(self.exp.device)
            transform["axis"] = self.exp.get_new_axis().to(self.exp.device)

        T = rm.RigidUnitQuat(
            linear=transform["quaternion"], translation=transform["centroid"]
        )

        fp = self.calculate_fine_points(T, self.exp.fine_shape).to(self.exp.device)
        img_fine = t.nn.functional.grid_sample(
            x, fp, mode="bilinear", align_corners=False
        )

        if self.exp.pass_features:
            features_fine = t.nn.functional.grid_sample(
                coarse["seg"], fp, mode="bilinear", align_corners=False
            )
            fine = self.unet_fine(t.cat((features_fine, img_fine), dim=1))
        else:
            fine = self.unet_fine(img_fine)

        return_dict = dict()
        return_dict["coarse"] = coarse["seg"]
        return_dict["transform"] = T
        return_dict["fine"] = fine["seg"]
        return_dict["axis"] = transform["axis"]
        return return_dict

    def get_affine_grid(self, shape: t.Tensor, transform: ty.Optional[t.Tensor] = None):
        """Calculate an affine grid, optionally transformed by a given tensor.

        Args:
            shape: Shape of the affine grid.
            transform: Optional transformation to apply to the affine grid.
        """
        if transform is None:
            transform = geometry.affine_identity(3, self.exp.batch_size)
        return (
            t.nn.functional.affine_grid(
                transform,
                [self.exp.batch_size, self.exp.channels, *shape],
                align_corners=False,
            )
            .half()
            .to(self.exp.device)
        )

    def calculate_fine_points(
        self, transform: rm.RigidUnitQuat, shape: t.Tensor, inverse: bool = False
    ):
        """Calculate the affine grid for the fine segmentation.

        Args:
            transform: Transform for the fine segmentation.
            shape: Shape of the fine segmentation.
            inverse: Whether to calculate the forward or reverse transform.
        """
        # The call to `flip` is requred due to somewhat counterintuitive behavior of `t.nn.functional.affine_grid`.
        # See comment in dv.geometry.transform_from_seg for further details.
        scale = (
            t.Tensor(np.array(self.exp.fine_shape) / self.exp.input_shape())
            .flip(0)
            .to(self.exp.device)
        )
        scale = t.cat((scale, t.ones((1)).to(self.exp.device)), dim=0)
        scale = scale.diag().to(self.exp.device)
        transform = transform.normalize().to_homogeneous().half().to(self.exp.device)
        if inverse:
            transform = t.matmul(scale.float().inverse().half(), transform)
        else:
            transform = t.matmul(transform, scale.half())
        return self.get_affine_grid(shape, transform[:, :3])

    def coarse_requires_grad(self, requires_grad: bool):
        """Method for freezing the coarse module.

        Args:
            requires_grad: If `False`, freeze the coarse module.
        """
        self.unet_coarse.set_requires_grad(requires_grad)

    def fine_requires_grad(self, requires_grad: bool):
        """Method for freezing the fine module.

        Args:
            requires_grad: If `False`, freeze the fine module.
        """
        self.unet_fine.set_requires_grad(requires_grad)

    def calculate_fully_connected_length(self):
        return (
            np.prod(
                [s // 2 ** len(self.exp.coarse_features) for s in self.exp.coarse_shape]
            )
            * self.exp.coarse_features[-1]
            * 2
        )


class UNet(t.nn.Module):
    """A basic UNet implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        features: list[int],
        norm: ty.Literal["group", "instance", "layer", "batch"],
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            features: List of feature lengths in each UNet layer.
            norm: Type of normalization.
        """
        super(UNet, self).__init__()
        self.dn_path = t.nn.ModuleList()
        self.up_path = t.nn.ModuleList()
        self.pool = layers.max_pool_dict[dimension](kernel_size=2, stride=2)

        # Downsampling path
        for ipt, opt in zip([in_channels] + features, features):
            self.dn_path.append(layers.CvNmAcxN(ipt, opt, dimension, norm=norm))

        # Upsampling path
        for feature in reversed(features):
            self.up_path.append(
                layers.conv_t_dict[dimension](
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.up_path.append(
                layers.CvNmAcxN(feature * 2, feature, dimension, norm=norm)
            )

        # Bottleneck
        self.bottleneck = layers.CvNmAcxN(
            features[-1], features[-1] * 2, dimension, norm=norm
        )

        # Final
        self.final_conv = layers.conv_dict[dimension](
            features[0], out_channels, kernel_size=1
        )

    def forward(self, x):
        skip_connections = []
        for down in self.dn_path:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        hook = x.clone()
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_path), 2):
            x = self.up_path[idx](x)
            skip_connection = skip_connections[idx // 2]

            # TODO: Allow inputs whose shapes aren't perfectly divisible
            # Could fix shape issues here:
            # if x.shape != skip_connection.shape:
            #    x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = t.cat((skip_connection, x), dim=1)
            x = self.up_path[idx + 1](concat_skip)

        return {"seg": self.final_conv(x), "hook": hook}

    def set_requires_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad = requires_grad

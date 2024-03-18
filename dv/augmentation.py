# System
import typing as ty

# Third Party
import itk
import numpy as np


def maybe_pass(f):
    """Decorator which runs the augmentation with a given probability, or otherwise returns the input unchanged."""

    def wrapper(*args):
        if np.random.uniform(0.0, 1.0) > args[0].probability:
            return args[1]
        else:
            return f(*args)

    return wrapper


class RandomGammaShift:
    """Class for applying a random gamma shift to the image."""

    def __init__(
        self,
        probability: float = 0.5,
        min_value: ty.Optional[float] = None,
        max_value: ty.Optional[float] = None,
        gamma_i_min=0.8,
        gamma_i_max=2.0,
    ):
        """Initialize the class.

        Args:
            probability: Probability of applying the transform.
            min_value: Minimum value of the input image range, used for normalization.
            max_value: Maximum value of the input image range, used for normalization.
            gamma_i_min: Lower bound of the (inverse) gamma correction coefficient range.
            gamma_i_max: Upper bound of the (inverse) gamma correction coefficient range.
        """
        self.probability = np.clip(probability, 0.0, 1.0)
        self.min_value = min_value
        self.max_value = max_value
        self.gamma_i_min = gamma_i_min
        self.gamma_i_max = gamma_i_max

    @maybe_pass
    def __call__(self, image):
        """Get an augmented version of the input image."""
        mn = self.min_value if self.min_value is not None else np.min(image)
        mx = self.max_value if self.max_value is not None else np.max(image)
        gamma_i = np.random.uniform(self.gamma_i_min, self.gamma_i_max)
        image = (image - mn) / (mx - mn)
        image = image ** (1 / gamma_i)
        image = image * (mx - mn) + mn
        return image


class RandomPosterize:
    """Class for image posterization (i.e. bit-depth reduction)."""

    def __init__(
        self,
        probability: float = 0.5,
        min_value: ty.Optional[float] = None,
        max_value: ty.Optional[float] = None,
        min_levels: int = 8,
        max_levels: int = 12,
    ):
        """Initialize the class.

        Args:
            probability: Probability of applying the transform.
            min_value: Minimum value of the input image range, used for normalization.
            max_value: Maximum value of the input image range, used for normalization.
            min_levels: Lower bound of the range of posterization levels.
            max_levels: Upper bound of the range of posterization levels.
        """
        self.probability = np.clip(probability, 0.0, 1.0)
        assert max_value > min_value, "max_value must be greater than min_value"
        self.min_value = min_value
        self.max_value = max_value
        assert min_levels > 1, "Cannot reduce the image to fewer than 2 levels."
        self.min_levels = min_levels
        assert max_levels > 1, "Cannot reduce the image to fewer than 2 levels."
        self.max_levels = max_levels

    @maybe_pass
    def __call__(self, image):
        """Get an augmented version of the input image."""
        levels = np.random.choice(range(self.min_levels, self.max_levels + 1))
        mn = self.min_value if self.min_value is not None else np.min(image)
        mx = self.max_value if self.max_value is not None else np.max(image)
        image = (image - mn) / (mx - mn) * levels
        image = image.astype(int).astype(float) / levels
        image = image * (mx - mn) + mn
        return image


class RandomPoissonNoise:
    """Class for addition of random Poisson noise."""

    def __init__(
        self,
        probability: float = 0.5,
        lam_min: float = 1.0,
        lam_max: float = 10.0,
        scale_min: float = 5.0,
        scale_max: float = 10.0,
    ):
        """Initialize the class.

        Args:
            probability: Probability of applying the transform.
            lam_min: Minimum lambda.
            lam_max: Maximum lambda.
            scale_min: Minimum scale.
            scale_max: Maximum scale.
        """
        self.probability = np.clip(probability, 0.0, 1.0)
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.generator = np.random.default_rng()

    @maybe_pass
    def __call__(self, image):
        """Get an augmented version of the input image."""
        lam = np.random.uniform(self.lam_min, self.lam_max)
        scale = np.random.uniform(self.scale_min, self.scale_max)
        noise = self.generator.poisson(lam=lam, size=image.shape) * scale - scale * lam
        return image + noise


def resample_isotropic_and_center(
    image,
    out_spacing: float,
    out_size: list[int],
    interpolate: ty.Literal["linear", "nearest"] = "linear",
    outside: float = 0,
    augmentation_seed: ty.Optional[int] = None,
):
    """Resample the input image about its center, optionally applying a spatial transformation.

    Args:
        out_spacing: Spacing of the output image.
        out_size: Matrix size of the output image.
        interpolate: Interpolation strategy.
        outside: Padding to use when interpolating outside the original image.
        augmentation_seed: Used to coordinate image/segmentation augmentations. If `None`, no augmentation is performed.
    """
    if np.isscalar(out_spacing):
        dimension = image.GetImageDimension()
        out_spacing = np.diag(np.asarray([out_spacing for _ in range(dimension)]))

    if interpolate == "linear":
        interpolator = itk.LinearInterpolateImageFunction.New(image)
    elif interpolate == "nearest":
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(image)
    else:
        assert (
            False
        ), f"interpolate = '{interpolate}' not recognized; allowed values are 'linear' and 'nearest'."

    if augmentation_seed is not None:
        t = bspline_transform_from_image(image, seed=augmentation_seed)
    else:
        t = itk.IdentityTransform[itk.D, 3].New()
    in_size = np.asarray(itk.size(image))
    in_origin = np.asarray(itk.origin(image))
    in_spacing = np.diag(np.asarray(itk.spacing(image)))
    in_direction = np.asarray(image.GetDirection())
    in_center_index = np.asarray([in_s // 2 for in_s in in_size])

    center = in_origin + np.dot(np.dot(in_direction, in_spacing), in_center_index)

    out_size = np.asarray(out_size)
    out_origin = center - np.dot(np.dot(in_direction, out_spacing), out_size / 2)

    image = itk.resample_image_filter(
        image,
        interpolator=interpolator,
        size=[int(os) for os in out_size],
        output_spacing=out_spacing.diagonal(),
        output_origin=out_origin,
        output_direction=image.GetDirection(),
        default_pixel_value=outside,
        transform=t,
    )

    return image


def bspline_transform_from_image(
    image,
    seed: ty.Optional[int] = None,
    scale: float = 0.1,
    rotation_cc_scale: float = np.pi / 8,
    rotation_ap_scale: float = np.pi / 16,
    rotation_lr_scale: float = np.pi / 16,
    translation_scale: float = 25.0,
    bspline_scale: float = 75.0,
):
    """

    Args:
        seed: Optional seed for a reproducible augmentation.
        scale: Defines a range of affine scales from `1.0 - scale` to `1.0 + scale`.
        rotation_cc_scale: Defines a range of affine rotations about the craniocaudal axis in radians.
        rotation_ap_scale: Defines a range of affine rotations about the anteroposterior axis in radians.
        rotation_lr_scale: Defines a range of affine rotations about the transverse axis in radians.
        translation_scale: Defines a range of translations in millimeters.
        bspline_scale: Scale for bspline control point displacement.
    """
    rand = np.random.default_rng(seed)

    # Image Info
    origin = np.array(itk.origin(image))
    spacing = np.array(itk.spacing(image))
    size = np.array(itk.size(image))
    size_half = size / 2
    fov = np.array([sp * sz for sp, sz in zip(spacing, size)])
    dimension = len(size)
    direction = image.GetDirection()
    center = origin + (size_half * spacing) @ np.array(direction)

    # B-Spline Transform
    tb_type = itk.BSplineTransform[itk.D, dimension, 3]
    tb = tb_type.New()
    tb.SetTransformDomainOrigin(origin)
    tb.SetTransformDomainPhysicalDimensions(fov)
    tb.SetTransformDomainMeshSize([3] * dimension)
    tb.SetTransformDomainDirection(direction)

    params = tb.GetParameters()
    for i in range(tb.GetNumberOfParameters()):
        params[i] = rand.uniform(-1.0, 1.0) * bspline_scale
    tb.SetParameters(params)

    # Affine Transform
    ta_type = itk.FixedCenterOfRotationAffineTransform[itk.D, dimension]
    ta = ta_type.New()
    ta.SetCenterOfRotationComponent(center)
    ta.Scale(rand.uniform(1.0 - scale, 1.0 + scale))
    ta.Rotate(0, 1, rand.uniform(-1.0, 1.0) * rotation_cc_scale)
    ta.Rotate(0, 2, rand.uniform(-1.0, 1.0) * rotation_ap_scale)
    ta.Rotate(1, 2, rand.uniform(-1.0, 1.0) * rotation_lr_scale)
    ta.Translate(rand.uniform(-1.0, 1.0, (3)) * translation_scale)

    # Composite Transform
    tc_type = itk.CompositeTransform[itk.D, dimension]
    tc = tc_type.New()
    tc.AddTransform(tb)
    tc.AddTransform(ta)

    return tc

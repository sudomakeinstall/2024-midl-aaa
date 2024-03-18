# System
import configparser as cp
import pathlib as pl
import itertools as it
import random
import typing as ty

# Third Party
import torch as t
import itk

itk.ProcessObject.SetGlobalWarningDisplay(False)
import numpy as np

# Internal
from . import utils
from . import geometry
from . import augmentation


class BasicDataset(t.utils.data.Dataset):
    def __init__(self, exp: utils.Experiment):
        """
        Initialize a dataset from an experiment class instantiation.

        A note on dimension ordering.  Once converted to a numpy array and/or pytorch tensor, the ordering of the data is [Caudal -> Cranial] x [Posterior -> Anterior] x [Right -> Left].  In order to provide a more uniform interface, certain ITK operations (specifically the `.shape` method and the bracket indexing) match the numpy/pytorch convention.  Note, however, that for internal operations this convention is reversed and this must be accounted for when making function calls.

        Args:
            exp: Experiment instance.
        """
        self.exp = exp
        series_trn = list(sorted(self.exp.dir_trn.glob("*/")))
        series_val = list(sorted(self.exp.dir_val.glob("*/")))
        series_tst = list(sorted(self.exp.dir_tst.glob("*/")))
        series = series_trn + series_val + series_tst

        self.series_partition = (
            [0] * len(series_trn) + [1] * len(series_val) + [2] * len(series_tst)
        )
        self.series_partition = np.array(self.series_partition)
        self.indices = np.array(range(len(series)))

        self.img_paths = [s / self.exp.img_file for s in series]
        self.seg_paths = [s / self.exp.seg_file for s in series]
        self.run_checks()

        self.augmentation = False
        self.gamma = augmentation.RandomGammaShift(
            min_value=self.exp.clip_min, max_value=self.exp.clip_max
        )
        self.posterize = augmentation.RandomPosterize(
            min_value=self.exp.clip_min, max_value=self.exp.clip_max
        )
        self.noise = augmentation.RandomPoissonNoise()

    def get_trn_indices(self):
        """Get the indices corresponding to the training set."""
        return self.indices[self.series_partition == 0]

    def get_val_indices(self):
        """Get the indices corresponding to the validation set."""
        return self.indices[self.series_partition == 1]

    def get_tst_indices(self):
        """Get the indices corresponding to the testing set."""
        return self.indices[self.series_partition == 2]

    def run_checks(self):
        """
        Ensure that all paths correspond to an existing file, and that there are an
        equal number of images and segmentations.
        """
        for i in self.img_paths:
            assert i.exists(), f"The file {i} doesn't exist."
        for s in self.seg_paths:
            assert s.exists(), f"The file {s} doesn't exist."
        assert len(self.img_paths) == len(
            self.seg_paths
        ), "The number of image and segmentation paths must be equal."

    def __len__(self) -> int:
        """
        Return the total number of datapoints.
        """
        return len(self.img_paths)

    def __getitem__(self, index: int):
        """Get a dictionary containing the image and segmentation for a given index.

        Args:
            index: Integer index to retrieve.
        """
        seed = random.randint(0, 100000) if self.augmentation else None
        img = self.load_img(self.img_paths[index], augmentation_seed=seed)
        seg = self.load_seg(self.seg_paths[index], augmentation_seed=seed)
        return {"img": img, "seg": seg}

    def load_img(
        self, path: pl.Path, augmentation_seed: ty.Optional[int] = None
    ) -> t.Tensor:
        """
        Read the input image, resample, perform augmentations, and normalize.

        Args:
            path: Path to the input image.
            augmentation_seed: Used to coordinate image/segmentation augmentations. If `None`, no augmentation is performed.

        Returns:
            An `(N+1)D` PyTorch tensor with channel dimension appended; i.e., if the resampled image has shape `[H, W, D]`, returned image will have shape `[1, H, W, X]`. Image is converted to half-precision `float16` prior to returning, but is left on the default device.
        """
        input_img = itk.imread(path).astype(np.float32)
        input_img = augmentation.resample_isotropic_and_center(
            input_img,
            self.exp.fine_spacing,
            self.exp.input_shape_itk(),
            interpolate="linear",
            outside=self.exp.air_hu,
            augmentation_seed=augmentation_seed,
        )
        input_img = itk.array_from_image(input_img)
        input_img = np.clip(input_img, self.exp.clip_min, self.exp.clip_max)

        if augmentation_seed is not None:
            input_img = self.gamma(input_img)
            input_img = self.posterize(input_img)
            input_img = self.noise(input_img)

        input_img = (input_img - np.mean(input_img)) / np.std(input_img)

        return t.from_numpy(input_img).half().unsqueeze(0)

    def load_seg(self, path: pl.Path, augmentation_seed: ty.Optional[int] = None):
        """Read the input segmentation, resample, and perform augmentations.

        Args:
            path: Path to the segmentation image.
            augmentation_seed: Used to coordinate image/segmentation augmentations. If `None`, no augmentation is performed.

        Returns:
            An `(N+1)D` PyTorch tensor with channel dimension appended; i.e., if the resampled image has shape `[H, W, D]`, returned image will have shape `[C, H, W, X]`. Image is converted to half-precision `float16` prior to returning, but is left on the default device.
        """
        seg = itk.imread(path).astype(np.uint8)
        seg = augmentation.resample_isotropic_and_center(
            seg,
            self.exp.fine_spacing,
            self.exp.input_shape_itk(),
            interpolate="nearest",
            augmentation_seed=augmentation_seed,
        )
        seg = itk.array_from_image(seg)
        # Remap the labels according to the dictionary from the experiment file.
        seg[np.isin(seg, list(self.exp.label_map.keys()), invert=True)] = 0
        for k, v in self.exp.label_map.items():
            seg[seg == k] = v
        seg = t.from_numpy(seg)
        seg = (
            t.nn.functional.one_hot(seg.long(), num_classes=self.exp.classes)
            .moveaxis(seg.dim(), 0)
            .half()
        )
        return seg

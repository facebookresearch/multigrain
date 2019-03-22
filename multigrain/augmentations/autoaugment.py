# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3

from enum import Enum, auto
from typing import Tuple, Any
from abc import ABC, abstractmethod

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

# from https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py

RGBColor = Tuple[int, int, int]
MIDDLE_GRAY = (128, 128, 128)


class ImageOp(Enum):
    SHEAR_X = auto()
    SHEAR_Y = auto()
    TRANSLATE_X = auto()
    TRANSLATE_Y = auto()
    ROTATE = auto()
    AUTO_CONTRAST = auto()
    INVERT = auto()
    EQUALIZE = auto()
    SOLARIZE = auto()
    POSTERIZE = auto()
    CONTRAST = auto()
    COLOR = auto()
    BRIGHTNESS = auto()
    SHARPNESS = auto()


class AutoAugmentPolicy(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, img: Any) -> Any:
        pass


class ImageNetPolicy(AutoAugmentPolicy):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor: RGBColor = MIDDLE_GRAY) -> None:
        self.policies = [
            SubPolicy(ImageOp.POSTERIZE, 8, 0.4, ImageOp.ROTATE, 9, 0.6, fillcolor),
            SubPolicy(
                ImageOp.SOLARIZE, 5, 0.6, ImageOp.AUTO_CONTRAST, 5, 0.6, fillcolor
            ),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.8, ImageOp.EQUALIZE, 3, 0.6, fillcolor),
            SubPolicy(ImageOp.POSTERIZE, 7, 0.6, ImageOp.POSTERIZE, 6, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.4, ImageOp.SOLARIZE, 4, 0.2, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 4, 0.4, ImageOp.ROTATE, 8, 0.8, fillcolor),
            SubPolicy(ImageOp.SOLARIZE, 3, 0.6, ImageOp.EQUALIZE, 7, 0.6, fillcolor),
            SubPolicy(ImageOp.POSTERIZE, 5, 0.8, ImageOp.EQUALIZE, 2, 1.0, fillcolor),
            SubPolicy(ImageOp.ROTATE, 3, 0.2, ImageOp.SOLARIZE, 8, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.6, ImageOp.POSTERIZE, 6, 0.4, fillcolor),
            SubPolicy(ImageOp.ROTATE, 8, 0.8, ImageOp.COLOR, 0, 0.4, fillcolor),
            SubPolicy(ImageOp.ROTATE, 9, 0.4, ImageOp.EQUALIZE, 2, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.0, ImageOp.EQUALIZE, 8, 0.8, fillcolor),
            SubPolicy(ImageOp.INVERT, 4, 0.6, ImageOp.EQUALIZE, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 4, 0.6, ImageOp.CONTRAST, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.ROTATE, 8, 0.8, ImageOp.COLOR, 2, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 8, 0.8, ImageOp.SOLARIZE, 7, 0.8, fillcolor),
            SubPolicy(ImageOp.SHARPNESS, 7, 0.4, ImageOp.INVERT, 8, 0.6, fillcolor),
            SubPolicy(ImageOp.SHEAR_X, 5, 0.6, ImageOp.EQUALIZE, 9, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 0, 0.4, ImageOp.EQUALIZE, 3, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 7, 0.4, ImageOp.SOLARIZE, 4, 0.2, fillcolor),
            SubPolicy(
                ImageOp.SOLARIZE, 5, 0.6, ImageOp.AUTO_CONTRAST, 5, 0.6, fillcolor
            ),
            SubPolicy(ImageOp.INVERT, 4, 0.6, ImageOp.EQUALIZE, 8, 1.0, fillcolor),
            SubPolicy(ImageOp.COLOR, 4, 0.6, ImageOp.CONTRAST, 8, 1.0, fillcolor),
        ]

    def __call__(self, img: Any) -> Any:
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self) -> str:
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(AutoAugmentPolicy):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor: RGBColor = MIDDLE_GRAY) -> None:
        self.policies = [
            SubPolicy(ImageOp.INVERT, 7, 0.1, ImageOp.CONTRAST, 6, 0.2, fillcolor),
            SubPolicy(ImageOp.ROTATE, 2, 0.7, ImageOp.TRANSLATE_X, 9, 0.3, fillcolor),
            SubPolicy(ImageOp.SHARPNESS, 1, 0.8, ImageOp.SHARPNESS, 3, 0.9, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 8, 0.5, ImageOp.TRANSLATE_Y, 9, 0.7, fillcolor),
            SubPolicy(
                ImageOp.AUTO_CONTRAST, 8, 0.5, ImageOp.EQUALIZE, 2, 0.9, fillcolor
            ),
            SubPolicy(ImageOp.SHEAR_Y, 7, 0.2, ImageOp.POSTERIZE, 7, 0.3, fillcolor),
            SubPolicy(ImageOp.COLOR, 3, 0.4, ImageOp.BRIGHTNESS, 7, 0.6, fillcolor),
            SubPolicy(ImageOp.SHARPNESS, 9, 0.3, ImageOp.BRIGHTNESS, 9, 0.7, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 5, 0.6, ImageOp.EQUALIZE, 1, 0.5, fillcolor),
            SubPolicy(ImageOp.CONTRAST, 7, 0.6, ImageOp.SHARPNESS, 5, 0.6, fillcolor),
            SubPolicy(ImageOp.COLOR, 7, 0.7, ImageOp.TRANSLATE_X, 8, 0.5, fillcolor),
            SubPolicy(
                ImageOp.EQUALIZE, 7, 0.3, ImageOp.AUTO_CONTRAST, 8, 0.4, fillcolor
            ),
            SubPolicy(
                ImageOp.TRANSLATE_Y, 3, 0.4, ImageOp.SHARPNESS, 6, 0.2, fillcolor
            ),
            SubPolicy(ImageOp.BRIGHTNESS, 6, 0.9, ImageOp.COLOR, 8, 0.2, fillcolor),
            SubPolicy(ImageOp.SOLARIZE, 2, 0.5, ImageOp.INVERT, 3, 0.0, fillcolor),
            SubPolicy(
                ImageOp.EQUALIZE, 0, 0.2, ImageOp.AUTO_CONTRAST, 0, 0.6, fillcolor
            ),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.2, ImageOp.EQUALIZE, 4, 0.8, fillcolor),
            SubPolicy(ImageOp.COLOR, 9, 0.9, ImageOp.EQUALIZE, 6, 0.6, fillcolor),
            SubPolicy(
                ImageOp.AUTO_CONTRAST, 4, 0.8, ImageOp.SOLARIZE, 8, 0.2, fillcolor
            ),
            SubPolicy(ImageOp.BRIGHTNESS, 3, 0.1, ImageOp.COLOR, 0, 0.7, fillcolor),
            SubPolicy(
                ImageOp.SOLARIZE, 5, 0.4, ImageOp.AUTO_CONTRAST, 3, 0.9, fillcolor
            ),
            SubPolicy(
                ImageOp.TRANSLATE_Y, 9, 0.9, ImageOp.TRANSLATE_Y, 9, 0.7, fillcolor
            ),
            SubPolicy(
                ImageOp.AUTO_CONTRAST, 2, 0.9, ImageOp.SOLARIZE, 3, 0.8, fillcolor
            ),
            SubPolicy(ImageOp.EQUALIZE, 8, 0.8, ImageOp.INVERT, 3, 0.1, fillcolor),
            SubPolicy(
                ImageOp.TRANSLATE_Y, 9, 0.7, ImageOp.AUTO_CONTRAST, 1, 0.9, fillcolor
            ),
        ]

    def __call__(self, img: Any) -> Any:
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self) -> str:
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(AutoAugmentPolicy):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor: RGBColor = MIDDLE_GRAY) -> None:
        self.policies = [
            SubPolicy(ImageOp.SHEAR_X, 4, 0.9, ImageOp.INVERT, 3, 0.2, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 8, 0.9, ImageOp.INVERT, 5, 0.7, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 5, 0.6, ImageOp.SOLARIZE, 6, 0.6, fillcolor),
            SubPolicy(ImageOp.INVERT, 3, 0.9, ImageOp.EQUALIZE, 3, 0.6, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 1, 0.6, ImageOp.ROTATE, 3, 0.9, fillcolor),
            SubPolicy(
                ImageOp.SHEAR_X, 4, 0.9, ImageOp.AUTO_CONTRAST, 3, 0.8, fillcolor
            ),
            SubPolicy(ImageOp.SHEAR_Y, 8, 0.9, ImageOp.INVERT, 5, 0.4, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 5, 0.9, ImageOp.SOLARIZE, 6, 0.2, fillcolor),
            SubPolicy(ImageOp.INVERT, 6, 0.9, ImageOp.AUTO_CONTRAST, 1, 0.8, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 3, 0.6, ImageOp.ROTATE, 3, 0.9, fillcolor),
            SubPolicy(ImageOp.SHEAR_X, 4, 0.9, ImageOp.SOLARIZE, 3, 0.3, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 8, 0.8, ImageOp.INVERT, 4, 0.7, fillcolor),
            SubPolicy(ImageOp.EQUALIZE, 5, 0.9, ImageOp.TRANSLATE_Y, 6, 0.6, fillcolor),
            SubPolicy(ImageOp.INVERT, 4, 0.9, ImageOp.EQUALIZE, 7, 0.6, fillcolor),
            SubPolicy(ImageOp.CONTRAST, 3, 0.3, ImageOp.ROTATE, 4, 0.8, fillcolor),
            SubPolicy(ImageOp.INVERT, 5, 0.8, ImageOp.TRANSLATE_Y, 2, 0.0, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 6, 0.7, ImageOp.SOLARIZE, 8, 0.4, fillcolor),
            SubPolicy(ImageOp.INVERT, 4, 0.6, ImageOp.ROTATE, 4, 0.8, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 7, 0.3, ImageOp.TRANSLATE_X, 3, 0.9, fillcolor),
            SubPolicy(ImageOp.SHEAR_X, 6, 0.1, ImageOp.INVERT, 5, 0.6, fillcolor),
            SubPolicy(ImageOp.SOLARIZE, 2, 0.7, ImageOp.TRANSLATE_Y, 7, 0.6, fillcolor),
            SubPolicy(ImageOp.SHEAR_Y, 4, 0.8, ImageOp.INVERT, 8, 0.8, fillcolor),
            SubPolicy(ImageOp.SHEAR_X, 9, 0.7, ImageOp.TRANSLATE_Y, 3, 0.8, fillcolor),
            SubPolicy(
                ImageOp.SHEAR_Y, 5, 0.8, ImageOp.AUTO_CONTRAST, 3, 0.7, fillcolor
            ),
            SubPolicy(ImageOp.SHEAR_X, 2, 0.7, ImageOp.INVERT, 5, 0.1, fillcolor),
        ]

    def __call__(self, img: Any) -> Any:
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self) -> str:
        return "AutoAugment SVHN Policy"


class SubPolicy(object):

    ranges = {
        ImageOp.SHEAR_X: np.linspace(0, 0.3, 10),
        ImageOp.SHEAR_Y: np.linspace(0, 0.3, 10),
        ImageOp.TRANSLATE_X: np.linspace(0, 150 / 331, 10),
        ImageOp.TRANSLATE_Y: np.linspace(0, 150 / 331, 10),
        ImageOp.ROTATE: np.linspace(0, 30, 10),
        ImageOp.COLOR: np.linspace(0.0, 0.9, 10),
        ImageOp.POSTERIZE: np.round(np.linspace(8, 4, 10), 0).astype(np.int),
        ImageOp.SOLARIZE: np.linspace(256, 0, 10),
        ImageOp.CONTRAST: np.linspace(0.0, 0.9, 10),
        ImageOp.SHARPNESS: np.linspace(0.0, 0.9, 10),
        ImageOp.BRIGHTNESS: np.linspace(0.0, 0.9, 10),
        ImageOp.AUTO_CONTRAST: [0] * 10,
        ImageOp.EQUALIZE: [0] * 10,
        ImageOp.INVERT: [0] * 10,
    }

    def __init__(
        self,
        operation1: ImageOp,
        magnitude_idx1: int,
        p1: float,
        operation2: ImageOp,
        magnitude_idx2: int,
        p2: float,
        fillcolor: RGBColor = MIDDLE_GRAY,
    ) -> None:

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img: Any, magnitude: int) -> Any:
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            ImageOp.SHEAR_X: lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            ImageOp.SHEAR_Y: lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            ImageOp.TRANSLATE_X: lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            ImageOp.TRANSLATE_Y: lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            ImageOp.ROTATE: lambda img, magnitude: rotate_with_fill(img, magnitude),
            ImageOp.COLOR: lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            ImageOp.POSTERIZE: lambda img, magnitude: ImageOps.posterize(
                img, magnitude
            ),
            ImageOp.SOLARIZE: lambda img, magnitude: ImageOps.solarize(img, magnitude),
            ImageOp.CONTRAST: lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            ImageOp.SHARPNESS: lambda img, magnitude: ImageEnhance.Sharpness(
                img
            ).enhance(1 + magnitude * random.choice([-1, 1])),
            ImageOp.BRIGHTNESS: lambda img, magnitude: ImageEnhance.Brightness(
                img
            ).enhance(1 + magnitude * random.choice([-1, 1])),
            ImageOp.AUTO_CONTRAST: lambda img, magnitude: ImageOps.autocontrast(img),
            ImageOp.EQUALIZE: lambda img, magnitude: ImageOps.equalize(img),
            ImageOp.INVERT: lambda img, magnitude: ImageOps.invert(img),
        }

        self.operation1 = func[operation1]
        self.magnitude1 = self.ranges[operation1][magnitude_idx1]
        self.p1 = p1
        self.operation2 = func[operation2]
        self.magnitude2 = self.ranges[operation2][magnitude_idx2]
        self.p2 = p2

    def __call__(self, img: Any) -> Any:
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


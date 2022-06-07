import cv2
import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import pad

"""
Contains the classes and functions to help use the classifier.
Currently has Transformer that transforms the image correctly to be passed
into the model.
"""


class SquarePad:
    def __init__(self, use=False):
        self.use = use

    def __call__(self, image):
        if not self.use:
            return image

        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 0, 'constant')


class ThreshMask:
    def __init__(self, use=False):
        self.use = use

    def __call__(self, image):
        if self.use:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
            mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 22)
            mask = cv2.bitwise_not(mask)
            image = cv2.bitwise_and(image, image, mask=mask)
        return image


class Transformer:
    def __init__(self, thresh_use=True, square_use=True, img_size=28) -> None:
        self.thresh_use = thresh_use
        self.square_use = square_use
        self.img_size = img_size

        self.transformer = transforms.Compose([
            ThreshMask(use=thresh_use),
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            SquarePad(use=square_use),
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
        ])

    def __call__(self, img) -> np.ndarray:
        return self.transformer(img)

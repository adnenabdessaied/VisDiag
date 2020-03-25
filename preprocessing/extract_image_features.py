#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import os
import cv2
import numpy as np
import torch
import copy

"""
        A class that will be used as a transform for our dataset.
"""


class ExtractImageFeatures(object):
    def __init__(self, resize_dim: (tuple, int)):
        """
        Class constructor.
        :param resize_dim: The dimension of images to fit our feature extractor (VGG16)
        """
        super(ExtractImageFeatures, self).__init__()
        assert isinstance(resize_dim, (tuple, int)) and len(resize_dim) == 2, "resize_dim must be of type tuple but" \
                                                                              "got {} instead".format(type(resize_dim))
        self.resize_dim = resize_dim

    def __call__(self, image_path, feature_extractor):
        """
        We make this class callable.
        :param image_path: The "absolute" path to the image.
        :param feature_extractor: The feature extractor used on images --> VGG16_clipped
        :return: a torch vector of image features
        """
        assert os.path.isfile(image_path), "There is no file under the given path {}".format(image_path)
        image = cv2.imread(image_path)
        # Convert BGR to RGB as cv2 reads color images in BGR format
        B, R = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
        image[:, :, 0], image[:, :, -1] = R, B
        image = cv2.resize(image, (self.resize_dim[0], self.resize_dim[1]))
        image = image / 255.0
        image = np.expand_dims(image, 0).astype(dtype=np.double)
        image = np.swapaxes(image, 1, 3)
        image = torch.from_numpy(image)
        feature_extractor.eval()
        with torch.no_grad():
            image_features = feature_extractor(image)
            # compute the l2-norm of the feature vector
            l2_norm = torch.norm(image_features, p=2, dim=1)
            image_features = image_features.div(l2_norm.expand_as(image_features))
        return image_features

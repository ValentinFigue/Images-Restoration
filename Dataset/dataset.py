import torch.utils
import numpy as np
from .parsing_tools import compute_image_paths, open_image
from .data_augmentation import crop_image
from .degradation import add_noise


class Dataset(torch.utils.data.Dataset) :
    """

    """

    def __init__(self, directory_path, training_mode, crop_size=48, noise_variance = 10):
        """

        Args:
            directory_path:
        """

        self.path_list = compute_image_paths(directory_path)
        self.training_mode = training_mode
        self.crop_size = crop_size
        self.noise_variance = noise_variance
        self.mean = [0.5,0.5,0.5]
        self.variance = [0.25,0.25,0.25]

    def __getitem__(self, item):
        """

        Args:
            item:

        Returns:

        """

        # Input reading
        image = open_image(self.path_list[item])
        # Data Augmentation
        if self.training_mode :
            image = crop_image(image, self.crop_size)
        # Conversion to PyTorch format
        degraded_image = add_noise(image,self.noise_variance)
        degraded_image = ((degraded_image/255)-self.mean)/self.variance
        image = ((image/255)-self.mean)/self.variance
        image = np.transpose(image,(2,0,1))
        degraded_image = np.transpose(degraded_image,(2,0,1))
        image = torch.FloatTensor(image)
        degraded_image = torch.FloatTensor(degraded_image)

        return image, degraded_image

    def __len__(self):

        return len(self.path_list)
import torch.utils
import numpy as np
from .parsing_tools import compute_image_paths, open_image
from .data_augmentation import crop_image
from .degradation import add_noise


class Dataset(torch.utils.data.Dataset):
    """
    Object representing a dataset which is used to train the network
    """

    def __init__(self, directory_path, training_mode, crop_size=48, noise_variance=10):
        """
        Initialization function
        Args:
            directory_path: Path to the folder containing the images to use as dataset.
            training_mode: Boolean to determine if the data augmentation is performed.
            crop_size: Size of the crop to use.
            noise_variance: Maximum variance of the noise which will be added to the images.
        """

        self.path_list = compute_image_paths(directory_path)
        self.training_mode = training_mode
        self.crop_size = crop_size
        self.noise_variance = noise_variance
        self.mean = [0.43110137, 0.43996549, 0.36798606]
        self.variance = [0.2103285, 0.1981421, 0.18789765]

    def __getitem__(self, item):
        """
        Overriding of the indexation function
        Args:
            item: Index in the dataset

        Returns:
            - A torch tensor corresponding to the input image
            - A torch tensor corresponding to the noised image
        """

        # Input reading
        image = open_image(self.path_list[item])
        image = image / 255
        # Data Augmentation
        if self.training_mode:
            image = crop_image(image, self.crop_size)
        # Conversion to PyTorch format
        degraded_image = add_noise(image, self.noise_variance / 255)
        degraded_image = (degraded_image - self.mean) / self.variance
        image = (image - self.mean) / self.variance
        image = np.transpose(image, (2, 0, 1))
        degraded_image = np.transpose(degraded_image, (2, 0, 1))
        image = torch.FloatTensor(image)
        degraded_image = torch.FloatTensor(degraded_image)

        return image, degraded_image

    def __len__(self):
        """
        Length operator
        Returns:
            The total number of images in the dataset.
        """

        return len(self.path_list)

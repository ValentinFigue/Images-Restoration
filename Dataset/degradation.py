import numpy as np


def add_noise(image, noise_variance):
    """
    Noise simulation function
    Args:
        image: Image on which to add the noise
        noise_variance: Maximum variance of the noise to add

    Returns:
        Numpy array of the noised image
    """

    noise = np.random.rand() * noise_variance * np.random.randn(*image.shape)
    image = np.clip(image + noise, 0, 1)

    return image

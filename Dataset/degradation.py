import numpy as np


def add_noise(image, noise_variance) :

    noise = np.random.rand()*noise_variance*np.random.randn(*image.shape)
    image = np.clip(image+noise,0,1)

    return image
import numpy as np


def add_noise(image, noise_variance) :

    noise = noise_variance*np.random.rand(*image.shape)
    image = np.clip(image+noise,0,255).astype(np.uint8)

    return image
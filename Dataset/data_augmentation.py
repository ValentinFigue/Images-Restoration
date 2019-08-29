import numpy as np


def crop_image(image, crop_size):
    """
    Crop function which is used to virtually augment the data
    Args:
        image: Numpy array of th image
        crop_size: Size of the crop to perform

    Returns:
        Numpy array of the cropped image
    """

    # Determine size of the input image
    height = image.shape[0]
    width = image.shape[1]

    # Determine the top left point of the crop
    indice_height = np.random.randint(0, height - crop_size)
    indice_width = np.random.randint(0, width - crop_size)

    return image[indice_height:indice_height + crop_size, indice_width:indice_width + crop_size]

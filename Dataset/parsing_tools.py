import os
from PIL import Image
import numpy as np


def compute_image_paths(directory_path):
    """
    Compute the paths for every images contained in the directory path

    Args:
        directory_path: Path containing the images to parse

    Returns:
        The list of the image to use

    """

    path_list = []
    # Loop over all the elements
    for root, dirs, files in os.walk(directory_path):
        for file in files :
            # Check that the file has a jpg or png extension
            extension = file[-4:]
            if extension in ['.png','.jpg'] :
                path_list.append(os.path.join(root,file))

    return path_list


def open_image(path):
    """
    Reading function

    Args:
        path: Path containing the image

    Returns:
        The image as numpy array

    """

    image = Image.open(path)
    image = np.array(image)

    return image

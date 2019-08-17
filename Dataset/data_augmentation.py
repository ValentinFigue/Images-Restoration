import numpy as np


def crop_image(image, crop_size) :

    height = image.shape[0]
    width = image.shape[1]

    indice_height = np.random.randint(0,height-crop_size)
    indice_width = np.random.randint(0,width-crop_size)

    return image[indice_height:indice_height+crop_size,indice_width:indice_width+crop_size]
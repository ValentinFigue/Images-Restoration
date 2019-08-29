import torch
import numpy as np


def inference(network, image, checkpoint_path):
    """
    Network inference function

    Args:
        network: Network object to make the inference
        image: Numpy array of the image to infer on
        checkpoint_path: Path to the weights to use for the inference

    Returns:
        Numpy array of the resulting image
    """

    # Weight loading
    network.load_state_dict(torch.load(checkpoint_path))

    # Image preparation
    mean = [0.43110137, 0.43996549, 0.36798606]
    variance = [0.2103285, 0.1981421, 0.18789765]
    image = ((image / 255) - mean) / variance
    image = np.transpose(image, (2, 0, 1))
    image = torch.FloatTensor(image)

    # Inference
    with torch.no_grad():
        output = network(image.unsqueeze(0))

    # Output transformation
    output_image = output.squeeze(0).numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = np.clip(255 * (output_image * variance + mean), 0, 255).astype(np.uint8)

    return output_image

import argparse
import torch
import numpy as np
from Dataset import open_image
from Network import RestorationNetwork


if __name__ == "__main__" :

    # Definition of the argument
    parser = argparse.ArgumentParser("Inference procedure script to use the restoration network on a specific image.")
    parser.add_argument("path", type=str, help="Path to the image to use the network on.")
    parser.add_argument("output_path", type=str, help="Path where the resulting image will be saved.")
    parser.add_argument("--checkpoint_path", type=str, help="Path where the resulting image will be saved.", default = 'Weight/noise.tar')
    args = parser.parse_args()

    image = open_image(args.path)
    network = RestorationNetwork()
    network.load_state_dict(torch.load(args.checkpoint_path))

    mean = [0.5, 0.5, 0.5]
    variance = [0.25, 0.25, 0.25]
    image = ((image / 255) - mean) / variance
    image = np.transpose(image, (2, 0, 1))
    image = torch.FloatTensor(image)

    output = network(image.unsqueeze(0))
    output_image = output.squeeze(0).numpy()
    output_image = np.clip(255*(output_image*variance+mean),0,255).astype(np.uint8)



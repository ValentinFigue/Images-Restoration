import argparse
from Network import RestorationNetwork, inference
from Dataset import open_image, export_image

if __name__ == "__main__":
    # Definition of the argument
    parser = argparse.ArgumentParser("Inference procedure script to use the restoration network on a specific image.")
    parser.add_argument("path", type=str, help="Path to the image to use the network on.")
    parser.add_argument("output_path", type=str, help="Path where the resulting image will be saved.")
    parser.add_argument("--checkpoint_path", type=str, help="Path where the resulting image will be saved.",
                        default='Weight/noise.tar')
    args = parser.parse_args()

    # Initialization of the network
    network = RestorationNetwork()
    # Import of the image
    image = open_image(args.path)
    # Inference
    output_image = inference(network, image, args.checkpoint_path)
    # Export of the image
    export_image(args.output_path, output_image)

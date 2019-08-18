import argparse
from Dataset import Dataset
from Network import RestorationNetwork, train_model


if __name__ == "__main__" :

    # Definition of the argument
    parser = argparse.ArgumentParser("Training procedure script to optimize the restoration network on a specific "
                                     "dataset.")
    parser.add_argument("path", type=str, help="Path to the directory containing the image to use for the Training.")
    parser.add_argument("output_path", type=str, help="Path to the directory where to save the models.")
    parser.add_argument("--learning_rate", type=int, help="Value of the learning rate to use.", default=0.0001)
    parser.add_argument("--weight_decay", type=int, help="Value of the weight decay to use.", default=0)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to optimize the network.", default=100)
    parser.add_argument("--batch_size", type=int, help="Size of the batch used during the training.", default=8)
    parser.add_argument("--num_workers", type=int, help="Number of CPU workers used to load data.", default=1)
    parser.add_argument("--crop_size", type=int, help="Size of the crop used during the training.", default=48)
    parser.add_argument("--max_noise_variance" , type = int, help = "Maximum variance noise used during the training.",
                        default = 50)
    parser.add_argument("--epoch_checkpoint", type=int, help="Number of epochs between two consecutive model exports.",
                        default=1)
    args = parser.parse_args()

    dataset = Dataset(args.path, True,args.max_noise_variance)
    if len(dataset)==0 :
        raise Exception("The folder provided does not contain any images.")
    network = RestorationNetwork()
    im, deg = dataset[0]
    train_model(network,dataset,args.output_path,args.num_epochs,args.batch_size,args.num_workers,args.learning_rate,args.weight_decay, args.epoch_checkpoint)

import argparse
from Dataset import Dataset
from Network import RestorationNetwork, train_model


if __name__ == "__main__" :

    # Definition of the argument
    parser = argparse.ArgumentParser("Training procedure script to optimize the restoration network on a specific "
                                     "dataset.")
    parser.add_argument("path", type=str, help="Path to the directory containing the image to use for the Training.")
    parser.add_argument("--learning_rate", type=int, help="Value of the learning rate to use.", default=0.0001)
    parser.add_argument("--weight_decay", type=int, help="Value of the weight decay to use.", default=0)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to optimize the network.", default=100)
    parser.add_argument("--batch_size", type=int, help="Size of the batch used during the training.", default=8)
    parser.add_argument("--num_workers", type=int, help="Number of CPU workers used to load data.", default=1)
    parser.add_argument("--crop_size", type=int, help="Size of the crop used during the training.", default=48)
    args = parser.parse_args()

    dataset = Dataset(args.path, True,)
    if len(dataset)==0 :
        raise Exception("The folder provided does not contain any images.")
    network = RestorationNetwork()
    train_model(network, dataset,args.num_epochs,args.batch_size,args.num_workers,args.learning_rate,args.weight_decay)

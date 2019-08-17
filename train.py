from Dataset import Dataset
from Network import RestorationNetwork, train_model

path = "/Users/valentinfigue/Desktop/Projets/TrainingVDSR"

dataset = Dataset(path, True)
network = RestorationNetwork()
train_model(network,dataset)


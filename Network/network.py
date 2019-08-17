import torch.nn as nn

class RestorationNetwork(nn.Module):

    def __init__(self, in_channels = 3):
        """
        Initialization of the network
        Args:
            in_channels: Number of channels of the input
        """

        super(RestorationNetwork,self).__init__()
        first_layer = nn.Sequential(nn.Conv2d(in_channels,64,3,1,1),nn.ReLU())
        intermediate_blocs = []
        for num in range(18):
            intermediate_blocs.append(nn.Conv2d(64,64,3,1,1))
            intermediate_blocs.append(nn.ReLU())
        intermediate_layers = nn.Sequential(*intermediate_blocs)
        last_layer = nn.Sequential(nn.Conv2d(64,in_channels,3,1,1))

        self.layers = nn.Sequential(first_layer, intermediate_layers, last_layer)

    def forward(self,x):
        """
        Forward function of the network definition

        Args:
            x: Torch tensor to run the network on

        Returns:
            Result of the inference

        """

        residual = x
        x = self.layers(x)
        x = residual + x

        return x
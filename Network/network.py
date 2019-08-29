import torch.nn as nn


class RestorationNetwork(nn.Module):
    """
    Object representing the network
    """

    def __init__(self, in_channels=3):
        """
        Initialization of the network function
        Args:
            in_channels: Number of channels of the input
        """

        super(RestorationNetwork, self).__init__()
        first_layer = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False), nn.ReLU())
        intermediate_blocs = []
        for num in range(18):
            intermediate_blocs.append(nn.Conv2d(64, 64, 3, 1, 1, bias=False))
            intermediate_blocs.append(nn.ReLU())
        intermediate_layers = nn.Sequential(*intermediate_blocs)
        last_layer = nn.Sequential(nn.Conv2d(64, in_channels, 3, 1, 1, bias=False))

        self.layers = nn.Sequential(first_layer, intermediate_layers, last_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
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

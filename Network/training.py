import time
import torch
import os


def train_model(model, dataset, output_path, num_epochs=25, batch_size=8, num_workers=1, learning_rate=0.0001,
                weight_decay=0, epoch_checkpoint=1):
    """
    Training function of the network

    Args:
        model: Network which need to be trained.
        dataset: Dataset to train the network on.
        output_path: Path where to store the trained coefficients.
        num_epochs: Total number of epochs.
        batch_size: Size of the batch of images to use during the training.
        num_workers: Number of CPU workers.
        learning_rate: Learning rate to use during the training.
        weight_decay: Weight decay to use during the training.
        epoch_checkpoint: Number of epochs between two consecutive savings of the model.

    """

    since = time.time()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=num_workers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()

    if not (os.path.exists(output_path)):
        os.makedirs(output_path)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        running_loss = 0.0

        # Iterate over data.
        for images, degraded_images in data_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            restored_images = model(degraded_images)
            loss = criterion(restored_images, images)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        print('Loss: {:.4f} \n'.format(epoch_loss))
        if ((epoch + 1) % epoch_checkpoint) == 0:
            torch.save(model.state_dict(), os.path.join(output_path, "checkpoint{}.tar".format(epoch + 1)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

import copy
import time
import torch


def train_model(model, dataset, num_epochs=25, batch_size = 8, num_workers = 1):

    since = time.time()
    val_acc_history = []

    data_loader = torch.utils.data.DataLoader(dataset,batch_size,True,num_workers=num_workers)
    criterion = torch.nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        # Iterate over data.
        for images, degraded_images  in data_loader:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            restored_images = model(degraded_images)
            loss = criterion(restored_images,images)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        print('Loss: {:.4f} '.format(epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
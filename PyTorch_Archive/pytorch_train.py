import torch
import os.path

# %% Train
"""This method trains the network. The accuracy, loss and network parameters are saved after each epoch
to avoid retraining networks. There is an option to continue training which loads from the file. """


def train(num_epochs, network, optimizer, loss_function, train_dataloader, save_path, continue_training=False):
    if continue_training:
        checkpoint = torch.load(save_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_accuracy = checkpoint['training_accuracy']
        train_loss = checkpoint['training_loss']
        begin_epochs = len(train_accuracy) + 1
        end_epochs = num_epochs + begin_epochs
    else:
        if os.path.isfile(save_path):
            raise Exception("File already exists. Please write to a new file.")
        train_accuracy = []
        train_loss = []
        begin_epochs = 1
        end_epochs = num_epochs + 1

    for epoch in range(begin_epochs, end_epochs):
        running_loss = 0
        running_accuracy = 0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = network(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += torch.sum(torch.max(output.data, 1)[1] == labels)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_accuracy.float().item() / len(train_dataloader.dataset)
        print("Epoch: ", epoch, " Loss: ", epoch_loss, " Accuracy: ", epoch_acc)
        train_accuracy.append(epoch_acc)
        train_loss.append(epoch_loss)

        torch.save({
            'training_accuracy': train_accuracy,
            'training_loss': train_loss,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)


# %% Test
def test(network, loss_function, test_dataloader, save_path):
    model = torch.load(save_path)
    network.load_state_dict(model['model_state_dict'])

    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            output = network(inputs)
            loss = loss_function(output, labels)
            test_loss += loss.item()
            test_accuracy += (torch.sum(torch.max(output.data, 1)[1] == labels)).float().item()
    test_loss /= len(test_dataloader.dataset)
    test_accuracy /= len(test_dataloader.dataset)

    model['testing_accuracy'] = test_accuracy
    model['testing_loss'] = test_loss

    torch.save(model, save_path)
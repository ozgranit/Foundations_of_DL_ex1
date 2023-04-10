import torch
import matplotlib.pyplot as plt

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from setup_and_baseline import get_data
from torch.utils.data import TensorDataset, DataLoader


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        hidden_size = 256
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10) # output size is 10, corresponding to the 10 CIFAR-10 classes

    def forward(self, x):
        x = x.view(-1, 32*32*3) # flatten input tensor
        x = torch.relu(self.fc1(x)) # apply ReLU activation function to hidden layer
        x = self.fc2(x) # output layer

        return x


def init_weights(net, std):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, 0)


def epoch_train(model, criterion, optimizer, train_loader):
    running_train_loss = 0.0
    running_train_acc = 0.0
    for inputs, labels in train_loader:  # returns batches
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_train_acc += torch.sum(preds == labels.data)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = running_train_acc / len(train_loader.dataset)

    return epoch_train_loss, epoch_train_acc


def epoch_test(model, criterion, test_loader):
    running_test_loss = 0.0
    running_test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:  # returns batches
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_test_acc += torch.sum(preds == labels.data)

    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    epoch_test_acc = running_test_acc / len(test_loader.dataset)

    return epoch_test_loss, epoch_test_acc


def train(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = epoch_train(model, criterion, optimizer, train_loader)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        epoch_test_loss, epoch_test_acc = epoch_test(model, criterion, test_loader)
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        # print(
        #     f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
        #     f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}')

    return train_losses, train_accs, test_losses, test_accs


def plot(train_losses, test_losses, train_accs, test_accs, file_name):

    # plot train and test losses
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.savefig(file_name + '_loss.png')
    plt.clf()

    # plot train and test accuracies
    plt.figure()
    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracies')
    plt.legend()

    plt.savefig(file_name + '_acc.png')
    plt.close('all')


def ff_net_baseline(cifar_folder_path):
    num_epochs = 50
    batch_size = 64


    train_images, train_labels, test_images, test_labels = get_data(cifar_folder_path)

    # Define the train and test loaders
    train_dataset = TensorDataset(torch.tensor(train_images, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_images, dtype=torch.float32),  torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Net()
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the hyperparameters to search over
    momentum_values = [0.1, 0.5, 0.9]
    lr_values = [0.001, 0.01, 0.1]
    init_std_values = [0.01, 0.1, 1.0]

    best_test_acc = 0
    best_params = {}

    for momentum in momentum_values:
        for lr in lr_values:
            for std in init_std_values:
                # Initialize the neural network
                init_weights(model, std)
                # Define the optimizer
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

                train_losses, train_accs, test_losses, test_accs = train(model, criterion, optimizer,
                                                                         train_loader, test_loader, num_epochs)
                plot(train_losses, test_losses, train_accs, test_accs, file_name=f"lr={lr}_std={std}_mom={momentum}")

                print(f'For lr={lr}_std={std}_mom={momentum}:\n'
                    f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, '
                    f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}\n')
                if test_accs[-1] > best_test_acc:
                    best_test_acc = test_accs[-1]
                    best_params['best_test_acc'] = best_test_acc
                    best_params['momentum'] = momentum
                    best_params['lr'] = lr
                    best_params['std'] = std

    print("best params are:")
    print(best_params)



if __name__ == '__main__':
    cifar_folder_path = Path(r"C:\Users\ozgra\PycharmProjects\Foundations_of_DL\cifar-10-batches-py")
    ff_net_baseline(cifar_folder_path)


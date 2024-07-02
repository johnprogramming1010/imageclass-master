#From slides
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model_linear import LinearNet
from model_cnn import Net
#Added remaining imports not shown on slides
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

def convert_to_20_classes(labels):
    """
    Function to convert CIFAR100 labels to 20 classes.
    Args:
    labels: List of CIFAR100 labels.
    Returns:
    List of 20 class labels.
    """
    # Mapping from fine to coarse labels
    fine_to_coarse = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                      3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                      6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                      0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                      5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                      16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                      10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                      2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                      16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                      18,  1,  2, 15,  6,  0, 17,  8, 14, 13]

    # Create a new list of labels
    new_labels = []
    # Convert each label to the new label
    for label in labels:
        new_labels.append(fine_to_coarse[label])

    return new_labels



def train(model_type, epochs, batch_size, lr, num_classes):
        
        """
        Function to train the model.
        Args:
        model_type: Type of the model to train ('cnn' or 'linear').
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate for the optimizer.
        Returns:
        train_accuracies: list to store accuracies for each epoch
        train_losses: List to store losses for each epoch
        validation_accuracies: List to store validation accuracies for each epoch
        validation_losses: List to store validation losses for each epoch
        """

        # Set timestamp for the filename
        timestamp = str(datetime.now().strftime("%Y%m%d-%H%M"))
        # Data augmentation and normalization for training
        transform = transforms.Compose([transforms.ToTensor()])

        #Load CIFAR100 training dataset
        trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
        # Split the training dataset into training and validation datasets
        # Calculate the number of samples to include in the training set (80%)
        train_size = int(0.8 * len(trainset))
        # grabs only the first 80% of the training dataset
        train_indices = list(range(train_size))  # Selecting the first 80% of the dataset
        # Create a SubsetRandomSampler to select samples for training
        train_sampler = SubsetRandomSampler(train_indices)
        # Create a DataLoader for training data
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
        
        # Split the dataset into training and validation sets
        # val_size = len(trainset) - train_size
        val_indices = list(range(train_size, len(trainset)))
        # Create a SubsetRandomSampler to select samples for validation
        val_sampler = SubsetRandomSampler(val_indices)
        # Create a DataLoader for validation data
        valloader = DataLoader(trainset, batch_size=batch_size, sampler=val_sampler)

        if model_type == 'cnn':
            # Initialize the cnn model
            net = Net(num_classes)
            # Set up tensorboard writer for cnn model
            writer = SummaryWriter('runs/CIFAR100_' + model_type + '_' + timestamp)
        elif model_type == 'linear':
            # Initialize the linear model
            net = LinearNet(num_classes)
            # Set up tensorboard writer for linear model
            writer = SummaryWriter('runs/CIFAR100_' + model_type + '_' + timestamp)
        else:
            raise ValueError("Invalid model type. Choose 'cnn' or 'linear'.")

        # Set up loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        # Set device to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_accuracies = [] # List to store accuracies for each epoch
        train_losses = [] # List to store losses for each epoch
        validation_accuracies = [] # List to store validation accuracies for each epoch
        validation_losses = [] # List to store validation losses for each epoch

        # Run the training loop
        for epoch in range(epochs):
            # Set nums to zero for each epoch to calculate the average loss and accuracy
            running_loss = 0.0
            correct = 0
            total = 0
            net.train()

            # Loop over the training dataset
            for i, data in enumerate(trainloader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Move inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward, backward, and optimize
                outputs = net(inputs)

                # Convert labels to 20 classes if num_classes is 20
                if (num_classes == 20):
                    # Convert labels to 20 classes
                    labels = torch.tensor(np.array(convert_to_20_classes(labels))).long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Adjust statistics
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate the average loss and accuracy
            train_loss = running_loss / len(trainloader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Logging to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)

            # Validation loop
            net.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():  # No need to compute gradients during validation
                for data in valloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)

                    # Convert labels to 20 classes if num_classes is 20
                    if (num_classes == 20):
                        # Convert labels to 20 classes
                        labels = torch.tensor(np.array(convert_to_20_classes(labels))).long()

                    val_loss += criterion(outputs, labels).item()

                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate the average validation loss and accuracy
            val_loss /= len(valloader)
            val_accuracy = val_correct / val_total
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            
            # Print statistics to terminal
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%},\n\t       Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}")
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        # Close the writer
        writer.close()
        # Save the model
        print('Finished Training')
        PATH = f"./data/model_{model_type}_{timestamp}_eps-{epochs}_lr-{lr * 1000:.0f}.pt"
        torch.save(net.state_dict(), PATH)

        return train_accuracies, train_losses, validation_accuracies, validation_losses

def main():
        '''
        Main function to train the model
        
        Args:
        None
        
        Returns:
        None
        '''
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Training script for CIFAR100 classification')
        # assign which model to use
        parser.add_argument('--model_type', type=str, choices=['cnn', 'linear'], default='cnn', help='Type of model to use (cnn or linear)')
        # assign a value for epoch
        parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
        # assign a value for batch size
        parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training')
        # assign a value for learning rate 
        parser.add_argument('-l', '--lr', type=float, default=0.005, help='Learning rate')
        # assign a value for num_classes
        parser.add_argument('-n', '--num_classes', type=int, default=100, help='Number of classes in the dataset')
        args = parser.parse_args()

        train_accuracies, train_losses, validation_accuracies, validation_losses = train(args.model_type, args.epochs, args.batch_size, args.lr, args.num_classes)

        plt.figure(figsize=(12, 6))

        # Plot training accuracies
        plt.subplot(2, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_accuracies, label='Training Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training losses
        plt.subplot(2, 2, 2)
        plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot validation accuracies
        plt.subplot(2, 2, 3)
        plt.plot(range(1, args.epochs + 1), validation_accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot validation losses
        plt.subplot(2, 2, 4)
        plt.plot(range(1, args.epochs + 1), validation_losses, label='Validation Loss')
        plt.title('Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        print('Showing rates, close to continue...')

        plt.tight_layout()  # Adjust subplot parameters to give specified padding
        plt.show()
        print('Program Finished')

if __name__ == '__main__':
    # Call main function
    main()
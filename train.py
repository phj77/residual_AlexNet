import torch
import torchvision
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from alexnet import *
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset

if __name__ == "__main__":
    #hyperparameter###############################################
    batch = 250
    learning_rate = 0.001
    epoch = 30
    ##############################################################

    #model_dir
    model_dir = "./model_weights/alexnet_cifar10_skipconnection_BN_1.pth"

    #device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #setup model
    #Alexnet -> original AlexNet
    #Alexnet_BN -> AlexNet with Batch normalization
    #Alexnet_SC_BN -> skip connection with Batch normalization
    model = AlexNet_SC_BN(num_classes=10).to(device)


    #reproducibility
    torch.manual_seed(123)
    generator = torch.Generator().manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)

    #data argumentation and load
    train_trans = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    val_test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trans)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=val_test_trans)

    split = int(0.8 * len(train_dataset))
    train_indices = list(range(len(train_dataset)))[:split]
    val_indices = list(range(len(train_dataset)))[split:]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    #setup loader
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

    
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #function to train one epoch.
    def train_one_epoch(model, train_loader, optimizer, device):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss/total
        train_acc = correct/total
        return train_loss, train_acc

    #function to calculate validation accuracy
    def validate(model, val_loader, device):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        val_loss = running_loss / total
        return val_loss, val_acc
    
    #Accuracy/loss list
    train_accuracies = []
    val_accuracies = []
    train_losses = []  
    val_losses = []  

    #training
    min_val_loss = 100
    for epo in range(epoch):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, device)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_dir)
            print(f'Model saved at epoch {epo + 1}')

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch: {epo+1}/{epoch}, batch size: {train_loader.batch_size}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')
    
    # draw plot 
    plt.figure(figsize=(12, 6))
    #accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epoch + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid()
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    print("training over")
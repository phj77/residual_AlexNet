import torch
import torchvision
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from alexnet import *
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,Subset
from sklearn.metrics import classification_report, confusion_matrix

# feature extractor는 마지막에 classirier를 사용한 Alexnet의 것을 그대로 사용하고, linear regressor만 훈련한다.

if __name__ == "__main__":
    #hyperparameter###############################################
    #model = adam
    batch = 250
    learning_rate = 0.001
    epoch = 40
    ##############################################################

    #model dir
    model_load_dir = "./model_weights/alexnet_cifar10.pth"
    model_save_dir = "./model_weights/alexnet_cifar10_regressor.pth"

    #device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    #setup model -> SKIP CONNECTION 선택
    #model = AlexNet_SC(num_classes=10).to(device)
    model = AlexNet_LR(num_classes=10).to(device)

    #model load
    state_dict = torch.load(model_load_dir, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    #가중치 고정할 모델 layer 선택; linear regressor만 학습한다.
    for param in model.features.parameters():
        param.requires_grad_ = False

    print(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


    #function to train one epoch.
    def train_one_epoch(model, train_loader, optimizer, device):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
        train_loss = running_loss/total
        return train_loss

    #function to calculate validation accuracy
    def validate(model, val_loader, device):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = F.mse_loss(outputs, labels)
                running_loss += loss.item() * images.size(0)
                total += labels.size(0)
        val_loss = running_loss / total
        return val_loss
    
    #loss list
    train_losses = []  
    val_losses = []  

    #training
    for epo in range(epoch):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_loss = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch: {epo+1}/{epoch}, batch size: {train_loader.batch_size}, '
              f'Train Loss: {train_loss:.4f}'
              f'Val Loss: {val_loss:.4f}'
              f'Time: {epoch_time:.2f}s')
    
    # 손실 그래프 그리기
    plt.figure(figsize=(8, 6))  # 한 개의 플롯만 필요
    plt.plot(range(1, epoch + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 모델 저장
    torch.save(model.state_dict(), model_save_dir)
    print("Model saved to alexnet_cifar10_regressor.pth")



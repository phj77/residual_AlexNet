import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from alexnet import * 
import numpy as np

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load test dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

# Load model
model = AlexNet_SC2(num_classes=10).to(device)
model.load_state_dict(torch.load('./model_weights/alexnet_cifar10_skipConnection2_1.pth' ))
model.eval()

# Extract features before the final classifier
features = []
labels = []

if __name__ == '__main__':
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass through the model until the second-to-last layer
            x = model.features(images)
            # x = model.avgpool(x)
            # x = torch.flatten(x, 1)

            #skip connection일때
            identity = model.conv_skip(x)
            x = model.main_branch(x)
            x += identity
            x = model.Relu(x)
            x = model.conv2(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            #Forward pass through classifier layers except the last one
            for layer in model.classifier[:-1]:
                x = layer(x)

            features.append(x.cpu())
            labels.append(target.cpu())

    # Concatenate all features and labels
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Define PCA function
    def pca(features, n_components=2):
   
        # Centering the data
        features_mean = np.mean(features, axis=0)
        features_centered = features - features_mean

        # Compute covariance matrix
        cov_matrix = np.cov(features_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Log top eigenvalues for debugging
        print(f"Top eigenvalues: {eigenvalues[:10]}")

        # Select the top n_components eigenvectors
        top_eigenvectors = eigenvectors[:, :n_components]

        # Project the data onto the top n_components eigenvectors
        features_reduced = np.dot(features_centered, top_eigenvectors)

        return features_reduced


    # Apply PCA to reduce features to 2D
    features_2d = pca(features, n_components=2)

    # Plot the 2D embedding space
    plt.figure(figsize=(10, 10))
    for i in range(10):
        idxs = labels == i
        plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=test_dataset.classes[i], alpha=0.6)

    for i in range(10):
        print(features_2d[i, 0], features_2d[i, 1])
    
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA of Feature Embeddings')
    plt.grid(True)
    plt.show()

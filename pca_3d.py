import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
model = AlexNet_SC3_BN(num_classes=10).to(device)
model.load_state_dict(torch.load('./model_weights/alexnet_cifar10_skipconnection_BN_1.pth', weights_only=True))
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
            x = torch.flatten(x, 1)

            features.append(x.cpu())
            labels.append(target.cpu())

    # Concatenate all features and labels
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Define PCA function
    def pca(features, n_components=3):
    
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

    # Apply PCA to reduce features to 3D
    features_3d = pca(features, n_components=3)

    # Plot the 3D embedding space
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        idxs = labels == i
        ax.scatter(features_3d[idxs, 0], features_3d[idxs, 1], features_3d[idxs, 2], label=test_dataset.classes[i], alpha=0.6)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('3D PCA of Feature Embeddings')
    ax.legend()
    plt.show()

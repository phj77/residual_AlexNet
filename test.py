import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from alexnet import * 


# Test function
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    top1_correct = 0
    top3_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float().to(device), labels.long().to(device)
            outputs = model(images)
            _, topk_preds = outputs.topk(3, dim=1)  # Top-k predictions

            # Top-1 Accuracy
            top1_preds = topk_preds[:, 0]
            top1_correct += top1_preds.eq(labels).sum().item()

            # Top-3 Accuracy
            top3_correct += sum([labels[i].item() in topk_preds[i].cpu().numpy() for i in range(len(labels))])

            all_preds.extend(top1_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    # Overall Accuracy, Top-1, Top-3 Accuracy
    overall_accuracy = top1_correct / total_samples
    top1_accuracy = top1_correct / total_samples
    top3_accuracy = top3_correct / total_samples

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)

    return overall_accuracy, top1_accuracy, top3_accuracy, cm, cr

if __name__ == "__main__":
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #model dir
    model_dir = "./model_weights/alexnet_cifar10_skipConnection2_1.pth"

    # Load test dataset
    val_test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_test_trans)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False, pin_memory=True, num_workers=2)

    # Load model
    model = AlexNet_SC2(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_dir))  # 저장된 모델 가중치를 불러옴
    print("Model loaded successfully.")

    # Test the model
    overall_accuracy, top1_accuracy, top3_accuracy, cm, cr = test_model(model, test_loader, device)

    # Print evaluation metrics
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)
    print(f"Overall Accuracy (Top-1): {overall_accuracy:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f}")

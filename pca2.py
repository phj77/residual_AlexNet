import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np

# 사용자의 alexnet 파일에서 모델 임포트
# (같은 디렉토리에 alexnet.py가 있어야 함)
from alexnet import * # ---------------------------------------------------------
# 1. 설정 및 전역 변수
# ---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
# Windows 환경 에러 방지를 위해 필요한 경우 0으로 설정하거나
# 아래 main 블록을 유지해야 함
NUM_WORKERS = 2 

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# ---------------------------------------------------------
# 2. Hook 함수 정의 (모델 내부 값을 가져오기 위함)
# ---------------------------------------------------------
features_after_model = []

def get_features_hook(module, input, output):
    # Hook이 걸린 레이어의 출력값(feature)을 복사하여 리스트에 저장
    # output이 튜플이면 output[0] 사용 등 조정 필요
    features_after_model.append(output.detach().cpu())

# ---------------------------------------------------------
# 3. Main 실행 블록 (Windows Multiprocessing 에러 방지 필수)
# ---------------------------------------------------------
if __name__ == '__main__':
    print(f"Device: {device}")
    
    # 1) 데이터셋 로드
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2) 모델 로드
    model = AlexNet_SC3_BN(num_classes=10).to(device)
    
    # 모델 가중치 로드 (경로가 정확한지 확인 필요)
    model.load_state_dict(torch.load('./model_weights/alexnet_cifar10_skipconnection_BN_1.pth'))
    
    model.eval()

    # 3) Hook 등록
    # 모델의 classifier 중 마지막 레이어 직전(보통 ReLU나 Dropout)의 출력을 가져옴.
    # 모델 구조에 따라 인덱스(-2)는 달라질 수 있으니 print(model)로 확인 권장.
    # 만약 에러가 난다면 model.features 등 다른 모듈에 등록 가능.
    handle = model.classifier[-2].register_forward_hook(get_features_hook)

    # 4) Feature Extraction 수행
    input_images_flattened = [] # 모델 들어가기 전 (Before)
    labels_list = []            # 정답 라벨

    print("Extracting features (limit 2000 images for visualization)...")
    
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            # 시각화가 너무 느려지지 않도록 2000개만 사용
            if i * BATCH_SIZE >= 2000:
                break
            
            images = images.to(device)
            
            # (A) 모델 순전파 (Forward) -> Hook이 자동으로 features_after_model에 저장
            model(images)
            
            # (B) 원본 이미지 저장 (Before PCA용)
            # (Batch, 3, 224, 224) -> (Batch, 3*224*224)로 1차원 평탄화
            input_images_flattened.append(images.cpu().view(images.size(0), -1))
            
            labels_list.append(target.cpu())

    # Hook 제거
    handle.remove()

    # 5) 데이터 병합 (List -> Numpy Array)
    X_before = torch.cat(input_images_flattened, dim=0).numpy()
    X_after = torch.cat(features_after_model, dim=0).numpy()
    y = torch.cat(labels_list, dim=0).numpy()

    print(f"Before Model Shape: {X_before.shape}") # (2000, 150528) 예상
    print(f"After Model Shape:  {X_after.shape}")  # (2000, 4096) 예상 (AlexNet 기준)

    # 6) PCA 수행 (Sklearn)
    print("Running PCA...")
    pca_before = PCA(n_components=2)
    X_before_2d = pca_before.fit_transform(X_before)

    pca_after = PCA(n_components=2)
    X_after_2d = pca_after.fit_transform(X_after)

    # 7) 시각화 (Matplotlib)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 왼쪽: 모델 통과 전 (Raw Pixels)
    for i in range(10):
        idxs = y == i
        axes[0].scatter(X_before_2d[idxs, 0], X_before_2d[idxs, 1], label=classes[i], alpha=0.5, s=15)
    axes[0].set_title("Before AlexNet (Raw Input)")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].grid(True)

    # 오른쪽: 모델 통과 후 (Embeddings)
    for i in range(10):
        idxs = y == i
        axes[1].scatter(X_after_2d[idxs, 0], X_after_2d[idxs, 1], label=classes[i], alpha=0.5, s=15)
    axes[1].set_title("After AlexNet (Learned Features)")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].grid(True)
    
    # 범례 표시
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
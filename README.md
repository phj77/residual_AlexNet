# Residual_AlexNet
Original AlexNet과 residual mappning을 이용한 AlexNet의 구현
## Project environment
Python Version: 3.12.5
Deep Learning: pytorch 2.4.1, torchvision 0.19.1, torchaudio 2.4.1 (CUDA 11.8)
Data Analysis: pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.1

## Experiment environment
### Data argumentation
데이터 크기를 256x256으로 키우고 random하게 좌우로 뒤집고 crop하였다. 또한, RGB 채널별로 정규화를 진행했다.

### hyperparameter
model = Adam, batch = 250, learning rate = 0.001, epoch = 40, loss = cross entropy

## Experiment 
### Original AlexNet
<img width="925" height="463" alt="Image" src="https://github.com/user-attachments/assets/d098b727-43be-4c4e-9d74-62274be2ec29" />

<img width="561" height="572" alt="Image" src="https://github.com/user-attachments/assets/81870d19-b61f-4531-a8a5-01dd2508d57b" />

#### PCA
1) 모델의feature extractor를 통과한 직후의 embedding space
<img width="364" height="366" alt="Image" src="https://github.com/user-attachments/assets/4335e247-ac87-48ba-b3cb-4c7db7c52eba" />

2) 모델의 classifier의 마지막 층을 통과하기 직전
<img width="532" height="525" alt="Image" src="https://github.com/user-attachments/assets/1f8a2fec-1939-48cb-82dc-52d583b22583" />

Classifier를 통해서, 모델이 데이터를 좀 더 잘 구분할 수 있게 되었음을 직관적으로 알 수 있다.

### AlexNet with skip connection but not batch normalization
<img width="1016" height="145" alt="Image" src="https://github.com/user-attachments/assets/579a4c73-9609-4526-8d90-86217255fb48" />

모델이 전혀 학습되지 않아서 batch normalizaiton을 추가하여 학습해보았다.

### AlexNet with batch normalization but not skip connection
<img width="940" height="466" alt="Image" src="https://github.com/user-attachments/assets/dd82dc18-a869-404e-93d8-5897654559d0" />

<img width="720" height="755" alt="Image" src="https://github.com/user-attachments/assets/7327560c-046a-46d6-b990-aaf374ab6693" />

<img width="940" height="473" alt="Image" src="https://github.com/user-attachments/assets/59151bfe-cf59-45ee-b4a7-f2c593df5aaa" />

### AlexNet with batch normaliztion and skip connection
<img width="940" height="474" alt="Image" src="https://github.com/user-attachments/assets/814ce035-9165-4064-992f-7daad38ca58b" />

<img width="660" height="727" alt="Image" src="https://github.com/user-attachments/assets/b1df33a2-85cf-46d5-931e-c5a1e5f57ec0" />

<img width="940" height="481" alt="Image" src="https://github.com/user-attachments/assets/b625c387-0037-47ce-9281-29e65a634865" />

## Conclusion
1. Batch normaliztion이 없는 skip connection을 만들면  AlexNet에서 제대로 학습할 수 없다.
2. Batch normalization은 약 top 1 accuracy의 약 4.5%의 성능 향상을 만든다.
3. Skip connection을 Batch normalization과 함께 사용하면 (1)의 결론과 달리 제대로 학습이 된다. 하지만, skip connection이 있을 때와 없을 때의 유의미한 validation loss의 감소나 학습 속도 증가가 관찰되지는 않는다.
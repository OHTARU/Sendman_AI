import torch

# 간단한 텐서를 생성하여 GPU로 전송하고 연산을 수행
x = torch.tensor([1.0, 2.0, 3.0]).to('cuda')
y = x * 2
print(y)
print(y.device)

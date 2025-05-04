import torch, torchaudio
print(torch.__version__)           # 应是 2.5.1
print(torch.version.cuda)          # '12.4'
print(torch.cuda.is_available())   # True
x = torch.rand(1, device='cuda')
print(x)
print("Hello World")
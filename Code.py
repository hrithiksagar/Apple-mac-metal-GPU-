# accessing apple metal GPU instead of CUDA
import torch
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model.to(device)

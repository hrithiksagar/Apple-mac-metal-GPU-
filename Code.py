# accessing apple metal GPU instead of CUDA
import torch

if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("MPS backend is not available.")
    
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device, "- Is now assigned to this device")

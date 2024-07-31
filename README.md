# Apple-mac-metal-GPU

this is the method to activate Apple Metal GPU for training Machine Learning models. instad of CUDA Apple Metal has its own MPS GPU. 

Code:
import torch

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

model.to(device)

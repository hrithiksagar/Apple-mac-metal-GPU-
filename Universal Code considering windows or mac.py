def get_device_type():
    import torch
    if torch.cuda.is_available():  
        return "cuda"
    else: 
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            return "mps"
        else:
            return "cpu"
    
device = get_device_type()
print(device)

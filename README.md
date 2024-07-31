# Apple-mac-metal-GPU

this is the method to activate Apple Metal GPU for training Machine Learning models. instad of CUDA Apple Metal has its own MPS GPU (Metal Performance Shaders)






Code:
```
import torch
if torch.backends.mps.is_available():
    print("MPS backend is available.") 
else:
    print("MPS backend is not available.")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device, "- Is now assigned to this device")
```

Universal code weather it is mac/windows 
```
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
```

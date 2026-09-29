import os 
import numpy as np 
import torch 
 
 
def load_token_array(path): 
    path = os.fspath(path) 
    if not os.path.exists(path): 
        raise FileNotFoundError(f"token file not found: {path}") 
 
    file_size = os.path.getsize(path) 
    if file_size % 2 != 0: 
        raise ValueError("token file has an odd number of bytes") 
 
    return np.memmap(path, mode="r", dtype=np.dtype("<u2")) 
 
 
def get_batch(dataset, batch_size, sequence_length, device, generator): 
    if batch_size <= 0: 
        raise ValueError("batch_size must be positive") 
    if sequence_length <= 0: 
        raise ValueError("sequence_length must be positive") 
    if len(dataset) <= sequence_length: 
        raise ValueError("dataset is too short for the requested sequence_length") 
 
    starts = torch.randint(0, len(dataset) - sequence_length, (batch_size,), generator=generator) 
 
    windows = [ 
        torch.from_numpy(dataset[start : start + sequence_length + 1].astype(np.int64)) 
        for start in starts.tolist() 
    ] 
    batch = torch.stack(windows) 
 
    x = batch[:, :-1].to(device) 
    y = batch[:, 1:].to(device) 
    return x, y
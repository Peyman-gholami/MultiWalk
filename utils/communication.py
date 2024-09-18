import torch

def pack(tensors):
    buffer = torch.cat([t.view(-1) for t in tensors])
    return buffer

def unpack(buffer, shapes):
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end
    return entries

def num_bytes(tensor):
    return tensor.nelement() * tensor.element_size()


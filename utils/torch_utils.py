import torch

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(use_gpu=False):
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    return device

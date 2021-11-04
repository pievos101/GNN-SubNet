import torch
from torch_geometric.data import Dataset
from torchvision import transforms

class GraphDataset(Dataset):
    
    def __init__(self, data):
        super(GraphDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        return self.data[idx]
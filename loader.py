from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import trange


class PlateDataset(Dataset):
    def __init__(self, pth_img, fn_label, device):
        super(PlateDataset, self).__init__()
        x = pd.read_csv(fn_label)
        self.labels = torch.tensor(x.values, dtype=torch.float32).to(device)
        self.length = len(x)
        self.pth = pth_img
        self.transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )
        print('loading data...')
        self.data = [self.transforms(Image.open(self.pth + '%d.jpg' % item)).float()
                     for item in trange(self.length // 10)]

    def __len__(self):
        return self.length // 10

    def __getitem__(self, item):
        # img = Image.open(self.pth + '%d.jpg' % item)
        # return self.transforms(img).float(), self.labels[item]
        return self.data[item], self.labels[item]


def get_loader(pth_img, fn_label, device, batch_size=32, shuffle=False):
    ds = PlateDataset(pth_img, fn_label, device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

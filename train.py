import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from loader import get_loader

import torch_directml as td
from optimizer import AdamOptimizer

from recognizer import Recognizer

# DEVICE = torch.device('cpu')
DEVICE = td.device()
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

# PATH_IMG = 'data/train/images_p/'
PATH_IMG = 'data/train/images/'
PATH_CSV = 'data/train/labels.csv'
MODEL_SAVE_PATH = 'model/0/'

model = Recognizer().to(DEVICE)

loader = get_loader(
    pth_img=PATH_IMG,
    fn_label=PATH_CSV,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = AdamOptimizer(model.parameters(), lr=LR)
best_loss = 1e6  # 大数保证第一次损失值

for epoch in range(EPOCHS):
    for data, labels in (e := tqdm(loader)):
        # data, labels = data.to(DEVICE), labels.to(DEVICE)
        data, labels = data.to(DEVICE), labels
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, labels)
        lv = loss.item()
        loss.backward()
        optimizer.step()
        e.set_postfix(
            {
                'epoch': epoch + 1,
                'loss': lv,
            }
        )

        if lv < best_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH + 'best.pth')
            print('Best model has been saved.')
            best_loss = lv

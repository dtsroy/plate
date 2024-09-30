import time

import cv2
import torch
import torchinfo
from PIL import Image
from torchvision import transforms

from recognizer import Recognizer
import torch_directml as td

# device = torch.device('cpu')
device = td.device()
r = Recognizer().to(device)
# r = Recognizer()
r.load_state_dict(state_dict=torch.load('model/0/best.pth', map_location=device))

# print(torchinfo.summary(
#     r,
#     (512, 512),
#     batch_dim=0,
#     col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'),
#     verbose=0
# ))

p = 'data/train/images/10.jpg'

with torch.no_grad():
    t = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )
    x = t(Image.open(p)).float()
    x = x.to(device)
    x = x.unsqueeze(0)

    t1 = time.time()
    y = r(x)
    print('time:', time.time() - t1)

    print(y := y.cpu().numpy()[0])
    img = cv2.imread(p)
    h, w = img.shape[:2]
    cv2.rectangle(
        img,
        (int(w * (y[0] - y[2] / 2)), int(h * (y[1] - y[3] / 2))),
        (int(w * (y[0] + y[2] / 2)), int(h * (y[1] + y[3] / 2))),
        color=(0, 0, 255),
        thickness=2
    )

    cv2.imshow('a', img)
    cv2.waitKey(0)

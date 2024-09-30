import cv2
from tqdm import trange

N = 10000
ORIGIN = 'data/train/images'
DIST = 'data/train/images_p'

for i in trange(N):
    img = cv2.imread(ORIGIN + '/plate_%.6d.jpg' % i, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(DIST + '/%d.jpg' % i, img)

##### test
# import time
# from PIL import Image
#
# x = 0
# for k in range(3):
#     t1 = time.time()
#     for i in range(1000):
#         # img = cv2.imread(DIST + '/%d.jpg' % i, cv2.IMREAD_GRAYSCALE)
#         img = Image.open(DIST + '/%d.jpg' % i)
#     x += time.time() - t1
# print(x / 3)

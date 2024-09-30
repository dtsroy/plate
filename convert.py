from tqdm import trange

N = 10000

with open(r'data/train/labels.csv', 'w+') as f:
    f.write('x,y,w,h\n')
    for i in trange(N):
        r = open('data/train/labels/plate_%.6d.txt' % i, 'r')
        s = r.read().split(' ')
        f.write(f'{s[1]},{s[2]},{s[3]},{s[4]}\n')
        r.close()

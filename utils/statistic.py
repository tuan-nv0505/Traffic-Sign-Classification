import json
from cProfile import label
from collections import defaultdict
import cv2
import numpy as np

with open('../dataset/tt100k/other.json', 'r') as file:
    data = json.load(file)


def get_mark():
    with open('../dataset/tt100k/marks/genlist.txt', 'r') as file:
        l = []
        for line in file.readlines():
            line = line.strip()
            line = line.replace('dataset/tt100k/marks/pad-all/', '')
            line = line.replace('.png', '')
            l.append(line)
    return l

def stat_objects(data):
    cate = defaultdict(int)
    for info in data.values():
        for c in info['objects']:
            cate[c['category']] += 1


    stat = defaultdict(list)
    for k, v in cate.items():
        stat[v].append(k)

    range = [
        (1, 10),
        (10, 20),
        (20, 50),
        (50, 100),
        (100, 250),
        (250, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000)
    ]
    stat = np.array(list(map(lambda x: (x[0], len(x[1])), stat.items())))
    print(f'Number of objects: {sum(cate.values())}')
    print(f'Number of labels: {len(cate)}')
    print('number of appearances | number of labels')
    for x in range:
        print(f'{x[0]} - {x[1]}: {np.sum(stat[(stat[:,0] >= x[0]) & (stat[:,0] < x[1])], axis=0)[1]}')


def size_bbox(data):
    x, y = [], []

    for info in data.values():
        for i, obj in enumerate(info['objects']):
            bbox = obj['bbox']
            xmin, ymin, xmax, ymax = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
            x.append(xmax - xmin)
            y.append(ymax - ymin)

    x = np.array(x).reshape(-1, 1)
    print(f'xmin: {np.min(x)}  xmax: {np.max(x)}')
    y = np.array(y).reshape(-1, 1)
    print(f'ymin: {np.min(y)}  ymax: {np.max(y)}')
    size = np.hstack((x, y))

    ranges = [
        ((2, 2), (4, 4)),
        ((4, 4), (16, 16)),
        ((16, 16), (32, 32)),
        ((32, 32), (64, 64)),
        ((64, 64), (128, 128)),
        ((128, 128), (256, 256)),
        ((256, 256), (512, 512))
    ]

    for (w_min, h_min), (w_max, h_max) in ranges:
        count = np.sum((size[:,0] >= w_min) & (size[:,0] < w_max) & (size[:,1] >= h_min) & (size[:,1] < h_max))
        print(f"Size: ({w_min},{h_min}) to ({w_max},{h_max}): {count} bounding box")

if __name__ == '__main__':
    stat_objects(data)
    size_bbox(data)
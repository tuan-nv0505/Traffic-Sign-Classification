import cv2
import json
import os

with open('../dataset/tt100k/train.json', 'r') as file:
    data = json.load(file)

def get_bbox(data, category):
    info = data[category]
    img_path = os.path.join('../dataset/tt100k', info['path'])
    img = cv2.imread(img_path)

    for i, obj in enumerate(info['objects']):
        bbox = obj['bbox']
        xmin, ymin, xmax, ymax = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])

        cropped = img[ymin:ymax, xmin:xmax]

        cv2.imshow(f'Object {i}', cropped)

if __name__ == '__main__':
    # get_bbox(build_dataset, '6873')
    get_bbox(data, '36')

    key = cv2.waitKey(0)

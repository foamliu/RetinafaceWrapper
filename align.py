import os

import cv2 as cv
import numpy as np
from retinaface.detector import detect_faces
from utils import select_significant_face, align_face


def resize(img):
    max_size = 600
    h, w = img.shape[:2]
    if h <= max_size and w <= max_size:
        return img
    if h > w:
        ratio = max_size / h
    else:
        ratio = max_size / w

    img = cv.resize(img, (int(round(w * ratio)), int(round(h * ratio))))
    return img


def draw_bbox(img_raw, bbox, landms):
    # show image
    text = "{:.4f}".format(bbox[4])
    b = list(map(int, bbox))
    cv.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cx = b[0]
    cy = b[1] + 12
    cv.putText(img_raw, text, (cx, cy),
               cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landms
    cv.circle(img_raw, (landms[0], landms[5]), 1, (0, 0, 255), 4)
    cv.circle(img_raw, (landms[1], landms[6]), 1, (0, 255, 255), 4)
    cv.circle(img_raw, (landms[2], landms[7]), 1, (255, 0, 255), 4)
    cv.circle(img_raw, (landms[3], landms[8]), 1, (0, 255, 0), 4)
    cv.circle(img_raw, (landms[4], landms[9]), 1, (255, 0, 0), 4)


if __name__ == '__main__':
    folder = 'data/Aaron Eckhart'
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    np.random.shuffle(files)

    for file in files:
        filename = os.path.join(folder, file)
        # print(filename)
        img = cv.imread(filename)

        if img is not None:
            img_raw = resize(img)
            bboxes, landmarks = detect_faces(img_raw, top_k=5, keep_top_k=5)
            i = select_significant_face(img.shape[:2], bboxes)
            bbox, landms = bboxes[i], landmarks[i]

            img = img_raw.copy()
            draw_bbox(img, bbox, landms)
            cv.namedWindow(file)  # Create a named window
            cv.moveWindow(file, 150, 30)
            cv.imshow(file, img)

            img = align_face(img_raw, [landms])
            cv.imshow('aligned', img)
            cv.waitKey(0)

    print('Completed!')

import os
import cv2 as cv
from retinaface.detector import detect_faces

if __name__ == '__main__':
    folder = 'data/Aaron Eckhart'
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    for file in files:
        filename = os.path.join(folder, file)
        # print(filename)
        img = cv.imread(filename)
        if img is not None:
            bboxes, landmarks = detect_faces(img)



    print('Completed!')
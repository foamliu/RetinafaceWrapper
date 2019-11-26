import time

import cv2 as cv

from retinaface.detector import detect_faces

if __name__ == '__main__':
    # testing begin
    image_path = "images/test.jpg"
    img_raw = cv.imread(image_path, cv.IMREAD_COLOR)

    start = time.time()
    bboxes, landmarks = detect_faces(img_raw)
    end = time.time()
    elapsed = end - start

    print('avg time: {:5f} ms'.format(elapsed * 1000))

    num_faces = bboxes.shape[0]

    # show image
    for i in range(num_faces):
        b = bboxes[i]
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv.putText(img_raw, text, (cx, cy),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        landms = landmarks[i]
        cv.circle(img_raw, (landms[0], landms[5]), 1, (0, 0, 255), 4)
        cv.circle(img_raw, (landms[1], landms[6]), 1, (0, 255, 255), 4)
        cv.circle(img_raw, (landms[2], landms[7]), 1, (255, 0, 255), 4)
        cv.circle(img_raw, (landms[3], landms[8]), 1, (0, 255, 0), 4)
        cv.circle(img_raw, (landms[4], landms[9]), 1, (255, 0, 0), 4)
    # save image

    # cv.imwrite('images/result.jpg', img_raw)
    cv.imshow('image', img_raw)
    cv.waitKey(0)

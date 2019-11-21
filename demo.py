import time

import cv2
from tqdm import tqdm

from retinaface.detector import detect
from retinaface.loader import load_model

if __name__ == '__main__':
    net = load_model()

    # testing begin
    image_path = "images/test.jpg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elapsed = 0.0
    num_runs = 1000
    for i in tqdm(range(num_runs)):
        start = time.time()
        dets = detect(net, img_raw)
        end = time.time()
        elapsed += end - start

    print('avg time: {:5f} ms'.format(elapsed / num_runs * 1000))

    # show image
    for b in dets:
        vis_thres = 0.6
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image

    name = "test.jpg"
    cv2.imwrite(name, img_raw)

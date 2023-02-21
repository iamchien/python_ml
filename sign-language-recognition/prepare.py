import cv2
from hand_detector import handDetector
import numpy as np
import math
import os

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    offset = 25
    imgSize = 256

    data_dir = "./data/H"

    counter = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        img, hands = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['hand']['bbox']

            new = lambda x, offset: x - offset if (x - offset > 0) else 0
            x = new(x, offset)
            y = new(y, offset)

            # cv2.rectangle(img, (x-offset, y-offset), (x+w+offset, y+h+offset), (0,255,0), 10)
            imgCrop = img[y:y+h+(2*offset), x:x+w+(2*offset)]        
            
            (h,w,_) = imgCrop.shape
            if h/w > 1:
                w = math.ceil(imgSize/h * w)
                imgCrop = cv2.resize(imgCrop, (w, imgSize))
            else:
                h = math.ceil(imgSize/w * h)
                imgCrop = cv2.resize(imgCrop, (imgSize, h))
            
            (h,w,_) = imgCrop.shape
            h_gap, w_gap = math.ceil((imgSize - h)/2), math.ceil((imgSize - w)/2)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgWhite[h_gap:h + h_gap, w_gap:w + w_gap] = imgCrop
            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)

        if key == ord("s"):
            counter += 1
            path = os.path.join(data_dir, f'Image_{counter}.jpg')
            cv2.imwrite(path, imgWhite)
            print(counter)
        elif key == ord('q'):
            cap.release()
            break

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cap.release()
        #     break

    cv2.destroyAllWindows()

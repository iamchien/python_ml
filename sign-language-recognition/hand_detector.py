import cv2
import mediapipe as mp
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        hands = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # print(len(self.results.multi_hand_landmarks))
            for id, handLms in enumerate(self.results.multi_hand_landmarks):
                hands.append({id: None})
                hand = {'hand': {'cx':[],'cy':[], 'bbox': None}}

                min_x, min_y = math.inf, math.inf
                max_x, max_y = -math.inf, -math.inf

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
                for lm in handLms.landmark:
                    h,w,_ = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h) # center point in terms of pixel

                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy

                    hand['hand']['cx'].append(cx)
                    hand['hand']['cy'].append(cy)

                hand['hand']['bbox'] = [min_x, min_y, max_x-min_x, max_y-min_y] # x, y, w, h
                hands[id] = hand

        return img, hands

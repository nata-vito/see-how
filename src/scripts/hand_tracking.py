from turtle import pos
import cv2 as cv
import mediapipe as mp
#from api_request import FireRise 

class handDetector():
    # Parameters
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        # To video streal False, any image true
        self.mode               = mode     

        # Maximum number of hands to detect         
        self.maxHands           = maxHands

        # Complexity of the hand landmark model
        self.modelComplexity    = modelComplexity

        # Minimum confidence value from the hand detection model
        self.detectionCon       = detectionCon

        # Minimum confidence value from the landmark-tracking model
        self.trackCon           = trackCon

        # Solutions from MediaPipe
        self.mpHands            = mp.solutions.hands
        self.hands              = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw             = mp.solutions.drawing_utils

        self.handFingers        = "0" # To transform list in a string
        self.fingers            = [] # To storing fingers
        self.side               = "" # Right/Left
        
    # To find hands in video
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # Getting landmarks position
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # Hand Labeling
            for id, hand_handedness in enumerate(self.results.multi_handedness):
                self.label = hand_handedness.classification[0].label

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw: cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
            
        return lmList

    # Transforming list in a string
    def deconstructionHand(self, tracking):
        tracking.handFingers = ""
        if tracking.label == "Left" or tracking.label == "Right":
            for i in range(len(tracking.fingers)):
                tracking.handFingers += str(tracking.fingers[i])
        return tracking.handFingers


    def handsLabel(pose, tracking, ids):

        tracking.fingers = []

        # There is a hand in the frame
        if len(pose) != 0:
            
            # Finding the hand's label
            if tracking.label == 'Left':
                # hand Thumb -> Left
                if pose[ids[0]][1] > pose[ids[0] - 1][1]:
                        tracking.fingers.append(1)
                else: 
                    tracking.fingers.append(0)

            elif tracking.label == 'Right':
                # hand Thumb -> Right
                if pose[ids[0]][1] < pose[ids[0] - 1][1]:
                        tracking.fingers.append(1)
                else: 
                    tracking.fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                # Check finger reference points to define hand is open or not
                if pose[ids[id]][2] < pose[ids[id] - 2][2]:
                    tracking.fingers.append(1)
                else: 
                    tracking.fingers.append(0)
            
            print(tracking.label, tracking.fingers)

            # Decosntruction of List to String
            tracking.handFingers = tracking.deconstructionHand(tracking)
            tracking.side        = tracking.label
        
        return tracking.handFingers

    # Send data to api 
    """ def sendToApi(tracking):
        api = FireRise("https://myhand-ff333-default-rtdb.firebaseio.com/", tracking.fingers)
        api.putData("mao", True, None, tracking.fingers)
 """

def videoCapture():
    # Camera capture
    cap         = cv.VideoCapture(0)
    i           = 0
    tracking    = handDetector(detectionCon=0.75, maxHands=2)
    # Hand landmarks 
    ids         = [4, 8, 12, 16, 20]

    # Verify camera errors
    if(cap.isOpened() == False):
        print("Error openning the video")

    while(cap.isOpened()):
        success, frame  = cap.read()
        # Hand's contour
        contour         = tracking.findHands(frame)
        pose            = tracking.findPosition(frame)
        i               += 1

        # Detection and decosntruction of List to String
        handDetector.handsLabel(pose, tracking, ids)

        if success:
            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(frame, "Ola", (50,50), font, 1, (0,0,255), 2)
            cv.imshow('Frame', frame)
            key = cv.waitKey(1)

            # Exit by user hand
            if tracking.handFingers == "01100":
                break 

            # Exit by user using keyboard
            if key == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()
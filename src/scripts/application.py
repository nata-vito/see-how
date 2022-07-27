import math 
import cv2 as cv
import numpy as np
import hand_tracking as ht

def videoCapture():
    # Camera capture
    cap         = cv.VideoCapture(0)
    i           = 0
    tracking    = ht.handDetector(detectionCon=0.75, maxHands=2)
    # Hand landmarks 
    ids         = [4, 8, 12, 16, 20]

    # Verify camera errors
    if(cap.isOpened() == False):
        print("Error openning the video")

    while(cap.isOpened()):
        
        success, frame  = cap.read()

        # Flip frame to correct predict
        frame = cv.flip(frame,1)

        # Hand's contour
        contour         = tracking.findHands(frame)
        pose            = tracking.findPosition(frame)
        i              += 1

        if len(pose) != 0:
            print(pose[4], pose[8])

            x1, y1 = pose[4][1], pose[4][2]                             # Thumb tip cicle
            x2, y2 = pose[8][1], pose[8][2]                             # Index tip circle
            cx, cy = (x1 + x2) // 2, (y1 + y2) //2                      # Middle circle

            cv.circle(frame, (x1, y1), 10, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (x2, y2), 10, (255, 0, 255), cv.FILLED)
            cv.line(frame, (x1, y1), (x2,y2), (255,0,255), 3)
            cv.circle(frame, (cx, cy), 10, (255, 0, 255), cv.FILLED)

            length = math.hypot(x2-x1, y2-y1)
            length = np.interp(length, [30, 230], [0, 100])

            if length < 20:
                cv.circle(frame, (cx, cy), 10, (0, 255, 0), cv.FILLED)
            print(length)

        # Detection and decosntruction of List to String
        tracking.handsLabel(pose, ids)
        num = tracking.labelText()

        if success:
            font     = cv.FONT_HERSHEY_COMPLEX
            left     = (50,50)
            right    = (380, 50)

            if tracking.countFingers > 0:
                if tracking.label == 'Left':
                    cv.putText(frame, num, left, font, 1, (0,0,255), 2)
                else:
                     cv.putText(frame, num, right, font, 1, (0,0,255), 2)

            cv.imshow('Frame', frame)
            key = cv.waitKey(1)

            # Exit by user hand
            """ if tracking.handFingers == "01100":
                break  """

            # Exit by user using keyboard
            if key == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    videoCapture()
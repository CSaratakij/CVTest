import numpy as np
import cv2
import imutils
import math

class Info:
    def __init__(self, x, y, trackColor):
        self.x = x
        self.y = y
        self.trackColor = trackColor
        self.trackColorHSV = (0, 0, 0)

    def getBounds(self):
        pixel = self.trackColorHSV
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])

        bounds = [lower, upper]
        return bounds


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0, 0, 0)
lineType               = 2

mouseX = 0
mouseY = 0

currentFrame = None

infoA = Info(0, 0, (0, 0, 0))
infoB = Info(0, 0, (0, 0, 0))

info = (infoA, infoB)
pickIndice = 0

def setInfo(index, x, y):
    global info

    bgr = currentFrame[y, x]
    color = tuple([int(x) for x in bgr])

    info[index].trackColor = color

    currentFrameHSV = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2HSV)
    bgrHSV = currentFrameHSV[y, x]
    colorHSV = tuple([int(x) for x in bgrHSV])

    info[index].trackColorHSV = colorHSV


def onMouseClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouseX, mouseY
        global pickIndice

        mouseX, mouseY = x, y
        setInfo(pickIndice, x, y)

        pickIndice += 1

        if pickIndice > 1:
            pickIndice = 0


def run():
    global currentFrame
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[ Webcam ]")
    print("Width : ", width)
    print("Height : ", height)

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouseClick)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        currentFrame = blurred

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        bounds = info[0].getBounds()

        mask = cv2.inRange(hsv, bounds[0], bounds[1])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        boundsB = info[1].getBounds()

        maskB = cv2.inRange(hsv, boundsB[0], boundsB[1])
        maskB = cv2.erode(maskB, None, iterations=2)
        maskB = cv2.dilate(maskB, None, iterations=2)

        contoursA, h = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contoursB, h = cv2.findContours(maskB.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.drawContours(frame, contoursA, -1, (0,255,0), 3)
        # cv2.drawContours(frame, contoursB, -1, (0,255,255), 3)

        previousWidth = 0
        maxContoursA = ()

        for i in range(len(contoursA)):
            x, y, w, h = cv2.boundingRect(contoursA[i])

            if previousWidth < w:
                previousWidth = w
                maxContoursA = contoursA[i]

        if len(maxContoursA) > 0:
            x, y, w, h = cv2.boundingRect(maxContoursA)

            halfWidth = int(w / 2)
            halfHeight = int(h / 2)

            centerX = x + halfWidth
            centerY = y + halfHeight

            cv2.circle(frame, (centerX , centerY), halfWidth, (255, 0, 0), 2)
            cv2.putText(frame, str((centerX, centerY)),
                (40, 35),
                font,
                fontScale,
                fontColor,
                lineType)

        previousWidthB = 0
        maxContoursB = ()

        for i in range(len(contoursB)):
            x, y, w, h = cv2.boundingRect(contoursB[i])

            if previousWidthB < w:
                previousWidthB = w
                maxContoursB = contoursB[i]

        if len(maxContoursB) > 0:
            x, y, w, h = cv2.boundingRect(maxContoursB)

            halfWidth = int(w / 2)
            halfHeight = int(h / 2)

            centerX = x + halfWidth
            centerY = y + halfHeight

            cv2.circle(frame, (centerX , centerY), halfWidth, (0, 0, 255), 2)
            cv2.putText(frame, str((centerX, centerY)),
                (40, 65),
                font,
                fontScale,
                fontColor,
                lineType)

        isAbleToShowLine = len(maxContoursA) > 0 and len(maxContoursB) > 0

        if isAbleToShowLine:
            x1, y1, w1, h1 = cv2.boundingRect(maxContoursA)
            x2, y2, w2, h2 = cv2.boundingRect(maxContoursB)

            halfOrgA = (x1 + int(w1 / 2), y1 + int(h1 / 2))
            halfOrgB = (x2 + int(w2 / 2), y2 + int(h2 / 2))

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)
            distance = round(distance, 2)

            lineThickness = 3
            cv2.line(frame, halfOrgA, halfOrgB, (0, 255, 0), lineThickness)

            cv2.putText(frame, 'Distance : ' + str(distance),
                (10, 100),
                font,
                fontScale,
                fontColor,
                lineType)
        else:
            totalDetect = 0

            if len(maxContoursA) > 0:
                totalDetect += 1

            if len(maxContoursB) > 0:
                totalDetect += 1

            if totalDetect < 1:
                cv2.putText(frame, 'Need to select two color for tracking',
                    (10, 130),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.rectangle(frame, (10, 20, 20, 20), info[0].trackColor, 2, cv2.FILLED, 0)
        cv2.rectangle(frame, (10, 50, 20, 20), info[1].trackColor, 2, cv2.FILLED, 0)

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


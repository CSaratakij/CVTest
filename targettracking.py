import cv2
import imutils
# import matplotlib.pyplot as plt

def run():
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cap = cv2.VideoCapture(0)
        # ret, frame = cap.read()

        target_image = cv2.imread("target/a.jpg")
        camera_frame = cv2.imread("target/sample.jpg")
        # camera_frame = frame

        target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        camera_frame_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)

        # rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # plt.imshow(target_image_gray)
        # cv2.imshow('Test', target_image_gray)

        orb = cv2.ORB_create()

        keypoint, description = orb.detectAndCompute(target_image_gray, None)
        keypoint_camera_frame, description_camera_frame = orb.detectAndCompute(camera_frame_gray, None)

        output = target_image.copy()
        output_camera_frame = camera_frame.copy()

        # Extract image features by ORB
        for marker in keypoint:
            output = cv2.drawMarker(output, tuple(int(i) for i in marker.pt), color = (0, 255, 0))

        for marker in keypoint_camera_frame:
            output_camera_frame = cv2.drawMarker(output_camera_frame, tuple(int(i) for i in marker.pt), color = (0, 255, 0))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        matches = bf.match(description, description_camera_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        threshold = 20
        isMatchAcceptable = len(matches) > threshold

        if isMatchAcceptable:
            output_camera_frame = cv2.drawMatches(target_image, keypoint,
                                                  camera_frame, keypoint_camera_frame,
                                                  matches[:threshold], 0, flags = 2)

        resizeOutput = imutils.resize(output_camera_frame, width = 600)
        cv2.imshow('Output', resizeOutput)
        # cv2.imshow('Output', output_camera_frame)

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()


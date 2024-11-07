from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # Accessing the center of the first detected face
        center = bboxs[0]["center"]
        # Drawing a circle at the center of the detected face
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    # Change waitkey to waitKey
    cv2.waitKey(1)

import cv2

# Play video
cap = cv2.VideoCapture("./output_videos/result.mp4")
ret, frame = cap.read()
while (1):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or ret is False:
        cap.release()
        cv2.destroyAllWindows()
        break
    cv2.imshow('frame', frame)

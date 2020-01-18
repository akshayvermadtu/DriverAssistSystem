import cv2
import time
# import winsound

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
oldTime = time.time()

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray += 20
    faces = face_cascade.detectMultiScale(gray)

    for(x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # img_item = "my-image.png"
        # cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) != 0:
            a = "Yes"
            oldTime = time.time()
            print("Fine")
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        else:
            print("Not Fine")
            if time.time() - oldTime > 1:
                # winsound.PlaySound("beep-04.wav", winsound.SND_ASYNC)
                print("Warning.................................")

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

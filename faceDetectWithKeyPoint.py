import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

cv2.namedWindow("test")
cap = cv2.VideoCapture(0)  # 加载摄像头录制
# cap = cv2.VideoCapture("test.mp4") #打开视频文件
success, frame = cap.read()

#font=cv2.InitFont(cv2.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
font = cv2.FONT_HERSHEY_SIMPLEX
while success:
    success, frame1 = cap.read()
    shape = frame1.shape
    frame = cv2.resize(frame1, (shape[1]//2, shape[0]//2), interpolation=cv2.INTER_CUBIC)
    size = frame.shape[:2]
    image = np.zeros(size, dtype=np.float16)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image, image)
    divisor = 8
    h, w = size
    minSize = (w // divisor, h // divisor)
    dets = detector(image, 1)
    '''''
    if len(dets) > 0:
        for faceRect in dets:
            #x, y, w, h = faceRect
            cv2.rectangle(frame1, (faceRect.left()*2,faceRect.top()*2), (faceRect.right()*2,faceRect.bottom()*2), (0, 255, 0), 2)
            shape = frame.shape

            text = str(((faceRect.left()+faceRect.right())/2 - shape[1]/2)*2) + ", "+ str(((faceRect.top()+faceRect.bottom())/2 - shape[0]/2)*2)
            cv2.putText(frame1, text, (faceRect.left()*2-5, faceRect.top()*2-5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # 锁定 眼和嘴巴
            #cv2.circle(frame, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))   # 左眼
            #cv2.circle(frame, (x + 3 * w //4, y + h // 4 + 30), min(w // 8, h // 8), (255, 0, 0))   #右眼
            #cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), (255, 0, 0))   #嘴巴
    '''''
    if len(dets) > 0:
        for faceRect in dets:
            #x, y, w, h = faceRect
            cv2.rectangle(frame1, (faceRect.left()*2, faceRect.top()*2), (faceRect.right()*2 ,faceRect.bottom()*2), (0, 255, 0), 2)
            shape = frame.shape
            shapes = predictor(frame, faceRect)
            if shapes.num_parts == 68:
                #0-35 is the face key points, 36-41 are right eye key points, 42-47 are left eyes key points
                for i in range(36, 48):
                    cv2.circle(frame1, (shapes.part(i).x * 2, shapes.part(i).y * 2), 2, (0, 0, 255), -1)
                for i in range(48, 68):
                    cv2.circle(frame1, (shapes.part(i).x*2, shapes.part(i).y*2), 2, (0, 125, 230), -1)
                for i in range(30, 36):
                    cv2.circle(frame1, (shapes.part(i).x*2, shapes.part(i).y*2), 2, (255, 0, 0), -1)

            #print(type(shapes))
            text = str(((faceRect.left()+faceRect.right())/2 - shape[1]/2)*2) + ", "+ str(((faceRect.top()+faceRect.bottom())/2 - shape[0]/2)*2)
            cv2.putText(frame1, text, (faceRect.left()*2-5, faceRect.top()*2-5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("test", frame1)
    #win.set_image(image)
    key = cv2.waitKey(20)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyWindow("test")
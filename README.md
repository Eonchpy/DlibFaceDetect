# DlibFaceDetect
this project aimed to use the open source library: DLIB to extract facial key points from camera inputs, and draw the rectangle which contain the faces.

for face detection:
I use 
detector = dlib.get_frontal_face_detector()
dets = detector(image, 1) #image is the input image(from camera)
dets contain rectangles which contain faces.

for facial key points:
first we need to put shape_predictor_68_face_landmarks.dat file in the same path as the .py
then we 
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
assign the facial key points predictor we want to use

at last we called 
shapes = predictor(frame, faceRect)

shapes.parts are the dlib.point array which contains all the key points
shapes.num_parts number of key points.
shape.part(i) (i<68 and i>=0) are the exact point.


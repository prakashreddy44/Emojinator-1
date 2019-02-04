import cv2 
import dlib
from imutils import  face_utils
import numpy as np
from mask import mask
from imutils.face_utils import FaceAligner
from keras.models import load_model
import os



CNN_MODEL = 'cnn_model_keras.h5'
SHAPE_PREDICTOR_68='shape_predictor_68_face_landmarks.dat'
shape_predictor_68=dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector=dlib.get_frontal_face_detector()
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)

cnn_model = load_model(CNN_MODEL)

emojis_folder = 'emojis/'
emojis = []
for emoji in range(len(os.listdir(emojis_folder))):
    print(emoji)
    emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png'))

#cap=cv2.VideoCapture('http://192.168.43.1:8080/video')  
cap=cv2.VideoCapture(0)
while True:
 _,img=cap.read() 
 real=img  
 img=cv2.resize(img,(600,400))
 gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
 faces=detector(gray)
 if len(faces)>0:
  face=faces[0]
  shape_68=shape_predictor_68(gray,face)
  shape=face_utils.shape_to_np(shape_68)
  
  masked_img=mask(shape,gray)
  masked=cv2.bitwise_and(gray,masked_img)
  face_m=fa.align(masked,gray,face)
  face_m=cv2.flip(face_m,1) 
  face_m=cv2.resize(face_m,(110,110))
  img = face_m
  img = np.array(img, dtype=np.float32)
  img = np.reshape(img, (1, 110, 110, 1))
  pred=cnn_model.predict(img)
  pred_probab = pred[0]
  pred_class = list(pred_probab).index(max(pred_probab))
  
  #img = blend(img, emojis[pred_class], (x, y, w, h))
  print(pred_class)
  cv2.namedWindow("pred",cv2.WINDOW_NORMAL)
  cv2.namedWindow("real",cv2.WINDOW_NORMAL)
  cv2.imshow('pred',emojis[pred_class])
  cv2.imshow('real',real)
  if cv2.waitKey(1)==ord('q'):
   cv2.destroyAllWindows()   
   cap.release()
   break
  

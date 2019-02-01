import cv2 
import dlib
from imutils import  face_utils
from mask import mask
from imutils.face_utils import FaceAligner

SHAPE_PREDICTOR_68='shape_predictor_68_face_landmarks.dat'
shape_predictor_68=dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector=dlib.get_frontal_face_detector()
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)

#img=cv2.imread('face.png')
cap=cv2.VideoCapture('http://192.168.43.1:8080/video')
data_path='./faces/'  
count=0
emoji=0
while True:
 _,img=cap.read() 
   
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
  cv2.imshow('face',face_m)

 msg='press c to capture face for emoji no. '+str(emoji)+' count '+str(count)
 cv2.putText(img,msg,(0,10),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0))
 cv2.imshow('real',img)
 if cv2.waitKey(1)==ord('c'):
     file_name_path = './faces/' + str(emoji) +'/'+str(count)+ '.jpg'
     cv2.imwrite(file_name_path, face_m)
     count=count+1
     """if cv2.waitKey(1)==ord('q'):
      cv2.destroyAllWindows()   
      cap.release()
      break"""
 if count==450:
  emoji=emoji+1
  count=0
  
     
 

     
     

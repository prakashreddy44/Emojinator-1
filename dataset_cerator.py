import cv2
import pickle
import os
import numpy as np
from sklearn.utils import shuffle

folder='faces'
images_labels=[]
for emoji_num in os.listdir(folder):
    for i in range(450):
        print('Loading emoji number '+emoji_num+' photo number '+str(i))
        img=cv2.imread(folder+'/'+emoji_num+'/'+str(i)+'.jpg',0)
        if np.any(img==None):
            continue
        images_labels.append((np.array(img, dtype=np.uint8),int(emoji_num)))
 
images_labels=shuffle(shuffle(shuffle(shuffle(shuffle(images_labels)))))
     
images=[]
labels=[]
for image,label in images_labels:
 images.append(image)
 labels.append(label)        
 
train_images=images[:int(5/6*len(images))]
print('length of the training image set is '+str(len(train_images)))
with open('train_test_data/train_images','wb') as f:
    pickle.dump(train_images,f)
    
train_labels=labels[:int(5/6*len(labels))]
print('length of the training label set is '+str(len(train_labels)))
with open('train_test_data/train_labels','wb') as f:
    pickle.dump(train_labels,f)  
    
test_images=images[int(5/6*len(images)):]
print('length of the test image set is '+str(len(test_images)))
with open('train_test_data/test_images','wb') as f:
    pickle.dump(test_images,f)
    
test_labels=labels[int(5/6*len(labels)):]
print('length of the test label set is '+str(len(test_labels)))
with open('train_test_data/test_labels','wb') as f:
    pickle.dump(test_labels,f)    
    

 
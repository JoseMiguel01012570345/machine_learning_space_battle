import cv2
import numpy as np
import random
import os
os.system('cls')

img = cv2.imread( 'map.jpg' , cv2.IMREAD_GRAYSCALE )

patch_x=100
patch_y=100
patch = np.ones((patch_x,patch_y) , dtype='uint8')

count=0
limit=100
while count < limit:
    x=random.randint(0,img.shape[0] - patch_x)
    y=random.randint(0,img.shape[1] - patch_y)
    
    for k in range(patch_x):
        for l in range(patch_y):
            patch[k,l] = img[ x + k , y + l ] 
            
    patch = np.where(patch > 128,1.0,0.0)
    
    cv2.imwrite(f'./patches/{count}.jpg', patch * 255.0 )
    count += 1
    
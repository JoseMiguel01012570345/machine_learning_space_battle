import numpy as np
import os
import cv2
from train_selector import use_model
import copy
import keras.models

os.system('cls')

num_models = 100

validation_path = './dataset_black_white/validation'

validation_img = []
for i in os.listdir(validation_path):
    validation_img.append( np.where(cv2.imread( validation_path + '/' + i , cv2.IMREAD_GRAYSCALE ) > 128 , 1.0 , 0.0 ))

count = 0
_input_ = None
model_list = [ keras.models.load_model( f'./model_pattern_recog/model_v{i}' )  for i in range( num_models )  ]
    
offset = 100
for img in validation_img:
    print('generating...') 
    _input_ = copy.deepcopy( img )
    for patch_row in range(int(img.shape[1] / offset)):
        for patch_column in range(int(img.shape[0] / offset)):
                
                sample = np.ones((100,100))
                for i in range( 100 ): # get patch
                    
                    k = patch_row * offset + i
                    if k >= img.shape[0]: break
                    
                    for j in range( 100 ):
                        l= patch_column* offset + j
                    
                        if l >= img.shape[0]: break
                        sample[ j , i  ] = img[l,k]
    
                sample = use_model( x=sample , model_list=model_list )
                
                for i in range( 100 ): # set patch
                    
                    k = patch_row * offset + i
                    if k >= img.shape[0]: break
                    
                    for j in range( 100 ):
                        l= patch_column * offset + j
                    
                        if l >= img.shape[0]: break
                        img[l,k] = sample[ j , i  ]
               
        
    count += 1
    print(count)
    cv2.imwrite( f'./gallery/{count}I.jpg' , img=np.where( _input_ * 255.0 > 128.0 , 255.0 , 0.0) ) # write input in gallery folder
    cv2.imwrite( f'./gallery/{count}O.jpg' , img=np.where(img* 255.0 > 128.0 , 255.0 , 0.0) ) # write ouput in gallery folder
    
# from keras import models
import numpy as np
import os
import cv2
import json

def use_model():
    
    test_dir = './img_map_dataset/'
    model = models.load_model('autoencoder_shape_1026.h5')
    
    for img in range( len(os.listdir(test_dir))):
        
        image = cv2.imread( test_dir + f'map{img}.jpg' , cv2.IMREAD_GRAYSCALE )
        image = image.astype(np.float32) / 255
        
        if image is None: continue
        
        # original
        # print(image.shape)
        # cv2.imshow( "original" , image )
        # cv2.waitKey(0)
        
        image = image.reshape( 1 , 1026 , 1026 )
        prediction =  model.predict(image)
        
        prediction = prediction.reshape( 1026 , 1026 )
        prediction = process_result(prediction=prediction)

        cv2.imshow( "prediction" , prediction )
        cv2.waitKey(0)
    
    pass

def process_result(prediction:np.array):
    
    for i in range(len(prediction)):
        for j in range( len(prediction[i])):
            
            if prediction[i,j] > 0.5:
                
                prediction[i,j] = 255
            else:
                prediction[i,j] = 0
                
        
    return prediction

def test():
    
    path = '../MapsSet/'
    
    data = []
    for img in range( len(os.listdir(path))):
        
        file = open(path + f"/map{img}.json")
        json_file:dict = json.load(file)
        json_file.pop('row')
        json_file.pop('column')
        data.append( json_file )

    pass

    for index,item_map in enumerate(data):
        
        number_black = 0
        for row in item_map:
            
            new = item_map[row].split(' ')
            
            if len(new) == 1: continue
            
            number_black +=len(new)
        
        print(f"map{index}: {number_black}" )
        

test()

# use_model()


from keras import models
import numpy as np
import os
import cv2
os.system('cls')

def use_model():
    
    test_dir = './base/'
    model = models.load_model('autoencoder_shape=1026x1026.h5')
    
    for img in range( len(os.listdir(test_dir))):
        
        image = cv2.imread( test_dir + f'{img}.jpg' , cv2.IMREAD_GRAYSCALE )
        image = cv2.resize( image , (  800 ,600 ) )
        
        if image is None: continue
        
        # image = add_gaussian_noise(image=image)
        img = image
        image = normalize(image)
        
        image = image.reshape( 1 , 800 * 600 )
        prediction =  model.predict(image)
        prediction = prediction.reshape( 600 , 800 )
        print(prediction.shape)
        prediction = adjust_img(prediction=prediction)
        
        cv2.imshow( "orgininal" , img )
        cv2.waitKey(0)

        cv2.imshow( "prediction" , prediction )
        cv2.waitKey(0)
    
    return prediction

def adjust_img(prediction:np.array):
    
    count = 0
    avg = (np.amax( prediction) + np.amin(prediction) )/ 2
    
    for i in range(len(prediction)):
            for j in range(len(prediction[i])):
    
                value = prediction[i,j]
                if value < avg:
                    prediction[i,j]=0
                else:
                    count += 1
                    prediction[i,j]=255

    print( count / ( 800 * 600 ) )
    
    return prediction

def add_gaussian_noise(image, sigma=15):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image: Grayscale image as a NumPy array.
    - sigma: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy image as a NumPy array.
    """
    
    row, col = image.shape
    gauss = np.random.normal(0, sigma, (row, col)).astype('uint8')
    noisy_image = image + gauss
    
    return noisy_image

def normalize(data):
    
    norm = np.ones(data.shape)
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            
            if  data[i,j] < 128:
                norm[i,j] = 0
                
    return norm

use_model()

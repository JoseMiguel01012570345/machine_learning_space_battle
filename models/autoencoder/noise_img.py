import json
import os
import cv2
import numpy as np

os.system('cls')

def convert_img_json( image , folder_path ):

    image = cv2.resize( image , ( 800 , 600 ) )
    
    print(f'normalizing {folder_path}')
    image = normalize_img( image )
    
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
     
    file = open( folder_path , 'w')
    json.dump( image , file , indent=4 )

    return

def normalize_img( data ):
    
    img = { }
    
    for i in range(len(data)):
        
        img[i] = ' '
        for j in range(len(data[i])):
            
            if  data[i,j] < 128:
                img[i] += f'{j} '
                data[i,j] = 0
    
    return img

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

def apply_noise():
    
    path = './base/'
    
    for index, item in enumerate(os.listdir(path)):
        
        img = cv2.imread(path + item , cv2.IMREAD_GRAYSCALE )
        
        folder_val_path = f'./dataset/val/dataset{index}/image{index}.json'
        
        for i in range(400):
            
            noisy_img= add_gaussian_noise( img )
            folder_train_path = f'./dataset/train/dataset{index}/image{i}.json'
            convert_img_json( folder_path=folder_train_path , image=noisy_img )
        
        convert_img_json(image=img , folder_path=folder_val_path )
        
    pass

apply_noise()
from data import load
import cv2
import numpy as np
import os

def build_img():

    img_path = "../generator/dataset_black_white/train/0.jpg"
    maps = load()
    
    # Open the image file
    img = cv2.imread(img_path)
    os.system("cls")    
    for key in maps:
        
        print(key)
        matrix = to_matrix( map_name=key ,map=maps)
        
        for i in range( np.size (img,0)):
            for j in range( np.size (img,1)):
                img[i,j] = matrix[i,j]

        cv2.imwrite(f"./img_map_dataset/{key}.jpg",img)
        
    return img

def to_matrix(map_name,map):
   
    matrix = np.full((1026, 1026),255)

    for row in map[map_name]:
        matrix[ row[0] , row[1] ] = 0
    
    return matrix

build_img()
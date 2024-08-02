import json
import cv2
import os
import numpy as np

class store_img_as_json:
    
    def __init__(self) -> None:
        pass
    
    def build_dataset_from_img(self):
                
        path_train = './dataset_black_white/train/'
        path_val = './dataset_black_white/validation/'
        
        count = 0
        for item in os.listdir( path_train ):
            
            os.system('cls')
            print( f"\033[1;32m loading ./dataset_black_white/train/{item}...\033[0m")
            
            img = cv2.imread( path_train + item , cv2.IMREAD_GRAYSCALE )
            
            self.store_img_in_json( img=img , index=count )
            
            count += 1
        
        count += 1
        for item in os.listdir( path_val ):
            
            os.system('cls')
            print( f"\033[1;32m loading ./dataset_black_white/validation/{item}...\033[0m")
            
            img = cv2.imread( path_val + item , cv2.IMREAD_GRAYSCALE )
            
            self.store_img_in_json( img=img , index=count )
            
            count += 1
    
        pass
        
    def store_img_in_json( self , img , index ): 
        
        path = "./dataset_img_json/"
            
        new_img_as_str = {}
        for i,row in enumerate(img):
            s=" "
            for j,column in enumerate(row):
                
                if not column:
                    s += f"{i}_{j} "
    
            new_img_as_str[i] = s 

        with open( f"{path}img{index}.json" , 'w') as file:
    
            # Serialize the data to JSON and write it to the file
            json.dump( new_img_as_str , file, indent=4)
    
        file.close()
        
        return
    
    def load_dataset_json(self): # load json
        
        path = './dataset_img_json/'
        
        list_dir=os.listdir( path=path )
        
        for item in list_dir:
            
            file_path = f'{path}{item}'
            
            # os.system('cls')
            print(f'\033[1;32m loading {file_path} \033[0m')
        
            reader = open( file_path ,'r')
            
            json_data = json.loads(reader.read())
            
            reader.close()
            
            yield self.tokenize_data( json_data )
    
    def tokenize_data( self ,data): # tokenize json
        
        sample = np.ones( (1026 , 1026) ) * 255
        
        for row in data:
            data_row = data[row].split(" ")[1:-1]
            
            for element in data_row:
                pair= str(element).split("_")
                i = int(pair[0])
                j = int(pair[1])
                sample[i,j] = 0
        
        return sample
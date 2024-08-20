from keras.models import load_model
import numpy as np
import pygame
import cv2
import os
os.system('cls')

def apply_action( patch , i , j , img ):
    
    action_count = 0
    rw = 1
    
    while action_count < int(upper_bound / 2) and rw > 0 : # apply 10 000 actions or apply actions till an action is bad
        
        patch = np.expand_dims(patch , axis=0) # bath size
        patch = np.expand_dims(patch , axis=-1) # channel
        actions = model(patch) 
        action = np.argmax(actions) # best reward action index
        rw = actions.numpy()[0][action] * upper_bound # reward

        action_values = action_list[ action ]
        
        value = action_values['value'] # pixel value
        row = action_values['row']
        column = action_values['column']
        
        patch = np.squeeze(patch) # remove batch size and channel
        patch[ column , row ] = value 
        
        img[ j + column  , i + row ] = value * 255.0
        
        surface = pygame.surfarray.make_surface(img) # build surface
        
        render( surface=surface ) # render surface
        print("rendering...")
        
        os.system('cls')
        print(f'number of actions applied over patch ==> {action_count}')
        
        action_count += 1
        
    return patch

def discover_map( img: np.array ):
    
    patch_x = 100
    patch_y = 100 
    patch_count = 0
    img = np.where( img > 128 , 255.0 , 0.0 )
    
    for i in range(img.shape[1] - patch_x): # pass for every pixel of the image picking a patch of size [patch_y] x [patxhx]
        for j in range(img.shape[0] - patch_y ):
            
            patch = np.ones( (100,100) )
            print(f'patch id ==> {patch_count}')
            for y in range(patch_y):
                for x in range(patch_x):
                    patch[ x , y ] = img[ j + x , i + y ]

            patch = np.where(patch > 128 , 1.0 , 0.0 ) # normalize patch
            patch = apply_action( patch=patch , i=i , j=j , img=img  )
            
            patch_count += 1
            print( f"patches processed ==> {patch_count}" )
            
            for y in range(patch_y): # paste changes to the original image
                for x in range(patch_x):
                    img[ j + x , i + y ] = patch[ x , y ] * 255.0
    
def render( surface ): # render with pygame
    
    surface =pygame.transform.scale( surface , new_size )
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            
            pygame.quit()
            return
        
    screen.fill((0, 0, 0))
    screen.blit(surface , (0, 0))
    pygame.display.update() 
    
model_path = './model_target'
model = load_model(model_path)

img = np.array([])
action_list = [ { 'value': 0.0 , 'row': i , 'column': j } for j in range(100) for i in range(100) ]
action_list.extend([ { 'value': 1.0 , 'row': i , 'column': j } for j in range(100) for i in range(100) ])
upper_bound = len(action_list)

new_size=( 1026 , 700 )
pygame.init()
screen = pygame.display.set_mode( new_size )

img_path ='./dataset_black_white/train/1.jpg' # image path
image =cv2.imread( img_path , cv2.IMREAD_GRAYSCALE )
discover_map( image )
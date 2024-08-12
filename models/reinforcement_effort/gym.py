import numpy as np
import pygame
import cv2
import os
import random
import math

os.system('cls')

class map_2d:

    def __init__(self, path) -> None:
        
        img = cv2.imread( path , cv2.IMREAD_GRAYSCALE )
        self.white_or_black(img=img)
        
        pygame.init()
        screen_width = 1026
        screen_height = 700

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        image_path = path
        image_surface = pygame.image.load(image_path)

        new_width = screen_width  # New width for the image
        new_height = screen_height  # New height for the image
        resized_image = pygame.transform.scale(image_surface, (new_width, new_height))
        array = np.array(pygame.surfarray.array3d(resized_image))
        
        self.sample = array
        self.reset()
        
        self.surface = pygame.surfarray.make_surface(array)

    sample = np.array([])
    observation_space = np.array( [ [ j for j in range(20)] for i in range(20) ] )
    action_space = np.concatenate( (np.array( [ [ { 'value': 255.0 , 'row':i , 'column': j  } for j in range(20) ] for i in range(20) ]) ,
                                    np.array( [ [ { 'value': 0.0 , 'row':i , 'column': j  } for j in range(20) ] for i in range(20) ] )),
                                    axis=0
                            ).flatten() 
    
    upper_bound = 1.0
    lower_bound = 0.0
    
    black = False
    last_action = 0
    
    def white_or_black( self , img: np.array ):
        
        img = np.where( img > 128 , 1.0 , 0.0 )
        blackness= np.sum(img)
        
        if blackness / ( img.shape[0] * img.shape[1] ) > .5:
            self.black = False
            return
        
        self.black = True
        
    def reset(self):
        
        '''
        returns random state
        
        '''
        self.patch_x_size = 20
        self.patch_y_size = 20
        
        self.col_axis = random.randint( 0 , int(self.sample.shape[0] / self.patch_x_size) - 1 )
        self.row_axis = random.randint( 0 , int(self.sample.shape[1] / self.patch_y_size) - 1 )
        
        arr_random = np.array( [ [ self.sample[self.col_axis * self.patch_x_size + j , self.row_axis * self.patch_y_size + i ] for j in range(self.patch_x_size) ] for i in range(self.patch_y_size)  ] ).astype('uint8')
        self.observation_space = np.where( arr_random > 128.0 , 1.0 , 0.0 )
        
        # self.show()    
        
        return self.observation_space.flatten() , None
    
    def show(self):
        cv2.imshow('kk',self.observation_space)
        cv2.waitKey(0)
        
    def step(self , actions ):
        
        '''
        return: reward, done, truncated
        
        '''
        rw , done , truncated = self.apply_action( actions )
        
        array = np.array(pygame.surfarray.array3d(self.surface))
        
        for i in range(self.patch_y_size):
            for j in range(self.patch_x_size):
                array[ self.col_axis * self.patch_x_size + i , self.row_axis * self.patch_y_size + j] = self.observation_space[j,i][0] * 255.0
        
        self.surface = pygame.surfarray.make_surface( array )
        self.render( surface=self.surface )
        
        return self.observation_space.flatten() , rw , done , truncated , None

    def render(self , surface ):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                pygame.quit()
                return
            
        self.screen.fill((0, 0, 0))
        
        self.noisy_surface = surface
        
        self.screen.blit(self.noisy_surface, (0, 0))
        
        pygame.display.update()

    def apply_action(self , actions ):

        '''
        action ={ 'w':int , 'h':int , 'value':int }
        
        '''
        actions_to_apply = []
        max_val = 0
        for index , action in enumerate(actions.tolist()):
            
            if max_val < action: max_val = action
            if action >= .8:
                actions_to_apply.append(index)
         
        print(max_val)

        if actions_to_apply.__contains__(math.nan) :
            print( "danger" )
            return -100000 , 0 , 1
        
        truncate = 1
        done = 0
        rw = -1
        for action in actions_to_apply:
            
            if self.observation_space[self.action_space[action]['column'], self.action_space[action]['row']][0] == self.action_space[action]['value']:
                rw =-action
                print( self.action_space[action] , action , f"reward:{rw}" )
                return  rw , 0 , 1
        
            self.observation_space[self.action_space[action]['column'], self.action_space[action]['row']] = self.action_space[action]['value']
            
            rw1 , done1 , truncate1 = self.reward( action=action )
            
            if truncate1 == 0:
                truncate = 0
            else:
                truncate += truncate1
                
            rw += rw1
            done += done1
        
        if done > 0 : done = 1
        if truncate > 0: truncate = 1
        
        return rw , done , truncate
    
    def reward(self , action ):
        
        print(self.action_space[action])
        
        if self.action_space[action]['value'] == 255.0 and not self.black :
            
            rw = action
            
            print( self.action_space[action] , action , f"reward:{ rw }" )
            return rw , 0 , 0
        
        # if self.action_space[action]['value'] == 1 and self.black :
        #     print( self.action_space[action] , action , f"reward:{-100000}" )
        #     return -100000 , 0 , 0
        
        rw = -action
        print( self.action_space[action] , action , f"reward:{rw}" )
        
        return rw , 0 , 0
    
def make():
    return map_2d('./dataset_black_white/train/600.jpg')
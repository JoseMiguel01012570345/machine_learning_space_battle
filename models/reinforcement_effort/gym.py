import numpy as np
import pygame
import cv2
import os
import random
import math

os.system('cls')

class map_2d:

    def __init__(self, path) -> None:
        
        self.init_var()
        
        img = cv2.imread( path , cv2.IMREAD_GRAYSCALE )
        self.white_or_black(img=img)
        
        self.write_in_file('','w')
        
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

    def init_var(self):
        
        self.sample = np.array([])
        
        self.patch_x_size = 100
        self.patch_y_size = 100
        
        self.observation_space = np.array( [ [ j for j in range(self.patch_x_size)] for i in range(self.patch_y_size) ] )
        self.action_space = np.concatenate( (np.array( [ [ { 'value': 1.0 , 'row':i , 'column': j  } for j in range(self.patch_x_size) ] for i in range(self.patch_y_size) ]) ,
                                        np.array( [ [ { 'value': 0.0 , 'row':i , 'column': j  } for j in range(self.patch_x_size) ] for i in range(self.patch_y_size) ] )),
                                        axis=0
                                ).flatten() 
        
        self.upper_bound = len(self.action_space.flatten()) - 1
        self.lower_bound = 0
        
        self.black = False
        self.col_axis = -1
        self.row_axis = 0
    
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
        
        if self.col_axis == self.sample.shape[0] - self.patch_x_size:
            
            self.col_axis = random.randint( 0 , self.sample.shape[0] - self.patch_x_size - 1 )
            self.row_axis = random.randint( 0 , self.sample.shape[1] - self.patch_y_size - 1 ) 
            
            if self.row_axis == self.sample.shape[1] - self.patch_y_size:
                self.col_axis = random.randint( 0 , self.sample.shape[0] - self.patch_x_size - 1 )
                self.row_axis = random.randint( 0 , self.sample.shape[1] - self.patch_y_size - 1 ) 
        else:
            self.col_axis += 1 
            
        new_sample = np.array( [ [ self.sample[self.col_axis + j , self.row_axis + i ][0] for j in range(self.patch_x_size) ] for i in range(self.patch_y_size)  ] ).astype('uint8')
        self.observation_space = np.where( new_sample > 128.0 , 1.0 , 0.0 ) # extract next patch from sample
        
        # self.show()
        
        return self.observation_space , None
    
    def show(self):
        cv2.imshow('kk',self.observation_space * 255.0 )
        cv2.waitKey(0)
    
    def write_in_file(self , text , mode='a'): # list of actions taken
        
        file = open("./list_actions.txt", mode )
        file.write(f"{text} \n")
    
    def step(self , action ):
        
        '''
        return: state, reward, done, truncated
        
        '''
        
        rw , done , truncated = self.apply_action( action )
        
        self.sample = np.array(pygame.surfarray.array3d(self.surface))
        
        # paste patch in sample to render
        for i in range(self.patch_y_size):
            for j in range(self.patch_x_size):
                self.sample[ self.col_axis + i , self.row_axis + j] = self.observation_space[j,i] * 255.0
        
        self.surface = pygame.surfarray.make_surface( self.sample )
        self.render( surface=self.surface )
        # self.show()
        
        return self.observation_space , rw , done , truncated , None

    def render(self , surface ):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                pygame.quit()
                return
            
        self.screen.fill((0, 0, 0))
        
        self.noisy_surface = surface
        
        self.screen.blit(self.noisy_surface, (0, 0))
        
        pygame.display.update()

    def apply_action(self , action ):

        '''
        action ={ 'w':int , 'h':int , 'value':int }
        
        '''

        if not (action >=0 and action <= self.upper_bound ):
            
            rw = -100000
            self.write_in_file(f"danger: {action} , reward:{rw}") # record action in file
            return rw , 0 , 1
        
        action = int(action)
        
        # check if action is useless
        if self.observation_space[self.action_space[action]['column'], self.action_space[action]['row']] == self.action_space[action]['value']:
            
            rw = 0.0
            self.write_in_file( f"{self.action_space[action]} {action} reward:{rw}" ) # record action in file
            return  rw , 0 , 1
        
        self.observation_space[self.action_space[action]['column'], self.action_space[action]['row']] = self.action_space[action]['value']
        
        return self.reward( action=action )
    
    def reward(self , action ):
        
        rw = 0
        middle = int(self.upper_bound / 2)
        if action == middle:
            rw = middle
            pass
        elif action < middle:
            rw = -1/(action - middle) * self.upper_bound
        elif action > middle:
            rw = -1/( (self.upper_bound - action) - middle) * self.upper_bound
        
        self.write_in_file( f"{self.action_space[action]} {action} reward:{rw}" )
        
        return rw , 0 , 0
    
def make():
    return map_2d('./dataset_black_white/train/1.jpg')
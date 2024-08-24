import numpy as np
import cv2
import os
import pygame 

os.system('cls')

class map_2d:

    def __init__(self) -> None:
        
        self.init_var()
        self.write_in_file('','w' )
        self.write_in_file('','w' , file_path='./sample_number' )
    
        
    def init_var(self):
        
        self.done = False
        self.best_rw = 0
        self.sample = np.array([])
        self.reset_time = 0
        
        self.patch_x_size = 100
        self.patch_y_size = 100
        
        self.observation_space = np.array( [ [ j for j in range(self.patch_x_size)] for i in range(self.patch_y_size) ] )
        self.action_space = np.concatenate((
                                            np.array( [ [ { 'value': 0.0 , 'row':i , 'column': j  } for j in range(self.patch_x_size) ] for i in range(self.patch_y_size) ] ) ,
                                            np.array( [ [ { 'value': 1.0 , 'row':i , 'column': j  } for j in range(self.patch_x_size) ] for i in range(self.patch_y_size) ] )
                                            ) , 
                                           axis=1
                                           ).flatten() 
        
        self.upper_bound = len(self.action_space.flatten()) - 1
        self.lower_bound = 0
        
        self.black = False
        self.col_axis = -1
        self.row_axis = 0
        
        self.patches = [ ]
        
        for patch in os.listdir('./patches'): # patches to target
            img = cv2.imread( './patches/' + patch , cv2.IMREAD_GRAYSCALE )
            self.patches.append( np.where(img > 128 , 1.0 , 0.0 ))
        
        self.len_sample = 0
        self.sample_list = []
        self.iterator = 1
        
        sample_path = './dataset_black_white/train' # list samples
        for sample in os.listdir(sample_path):
            
            if len(self.sample_list) == 300:break
            self.sample_list.append( cv2.imread(sample_path + '/'+ sample ) )
        
        self.len_sample = len(self.sample_list)
        self.similarity_queue = []
        
    def white_or_black( self ):
        
        img = np.where( self.sample > 128 , 1.0 , 0.0 )
        blackness= np.sum(img)
        
        if blackness / ( img.shape[0] * img.shape[1] ) > .5:
            self.black = False
            return
        
        self.black = True
    
    def next(self):
        
        self.iterator += 1
        self.done = False
        self.reset_time = 0
        
        self.write_in_file(f"sample ==>{ self.iterator }" , file_path='./sample_number')
        
        if self.iterator >= self.len_sample: self.iterator = 0
        
        self.sample = self.sample_list[ self.iterator ]
        self.white_or_black()
        
    def reset(self):
        
        '''
        returns random state
        
        '''
        
        window_x = 30
        window_y = 30
        
        self.reset_time +=1 
        
        self.best_rw = 0
        self.col_axis += window_x
        
        if self.col_axis >= self.sample.shape[0] - self.patch_x_size:
            
            self.col_axis = 0
            self.row_axis += window_y
        
            if self.row_axis >= self.sample.shape[1] - self.patch_y_size:
                self.done = True
                self.row_axis =0
                self.col_axis = 0
                
        new_sample = np.array( [ [ self.sample[self.col_axis + j , self.row_axis + i ][0] for j in range(self.patch_x_size) ] for i in range(self.patch_y_size)  ] ).astype('uint8')
        self.observation_space = np.where( new_sample > 128.0 , 1.0 , 0.0 ) # extract next patch from sample
        
        progress =f"progress in sample { self.iterator } ==>{ self.reset_time / int((self.sample.shape[0] * self.sample.shape[1])/(window_x * window_y)) * 100 }"
        
        # self.observation_space = np.zeros( self.observation_space.shape )
        
        return self.observation_space , progress , None
    
    def show(self):
        cv2.imshow('kk',self.observation_space * 255.0 )
        cv2.waitKey(0)
    
    def write_in_file(self , text , mode='a' , jump=True ,  file_path="./list_actions"): # list of actions taken
        
        file = open( file_path , mode )
        
        if jump:
            file.write(f"{text} \n")
        else:
            file.write(f"{text}")
            
    def step(self , action ):
        
        '''
        return: state, reward, done, truncated
        '''
        
        column = self.action_space[action]['column']
        row = self.action_space[action]['row']
        value = self.action_space[action]['value']
        
        rw , done , truncated , modified = self.apply_action( action )
        
        # apply action
        if modified: self.sample[ self.col_axis + column , self.row_axis + row][0] = value * 255.0
        
        # surface = pygame.surfarray.make_surface( self.sample )
        # self.render(surface=surface)
        # cv2.imwrite( './record.jpg' , self.observation_space * 255.0 )
        
        return self.observation_space , rw , done , truncated , None

    def render(self , surface ):
        
        surface =pygame.transform.scale( surface , self.new_size )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                
                pygame.quit()
                return
            
        self.screen.fill((0, 0, 0))
        self.screen.blit(surface , (0, 0))
        pygame.display.update()

    def apply_action(self , action ): # action ={ 'w':int , 'h':int , 'value':int }
        action = int(action)
        return self.reward( action=action )
    
    def reward(self , action ):
        
        column = self.action_space[action]['column']
        row = self.action_space[action]['row']
        value = self.action_space[action]['value']
        modified = False
        
        actual_value = self.observation_space[ column , row ]
        
        self.observation_space[ column , row ] = value
        
        best_match = 0
        
        for patch in self.patches: # best match by cosine similarity
            similarity = self.matrix_cosine_similarity(patch , self.observation_space)
            if similarity > best_match:
                best_match = similarity
        
        best_match *= self.upper_bound
        
        if value == 1.0 and self.black: best_match = - best_match
        if value == 0.0 and not self.black: best_match = - best_match
        
        if best_match < self.best_rw:
            self.observation_space[ column , row ] = actual_value
        
        else:
            self.best_rw = best_match
            modified = True
            
        # self.write_in_file( f"{self.action_space[action]} {action} reward:{ best_match }" ) # record action in file
        
        return best_match , self.done , 0 , modified
    
    def matrix_cosine_similarity( self,  matrix_a, matrix_b):
        
        """
        Calculate the cosine similarity between two matrix.
        
        Parameters:
        vector_a (numpy.ndarray): The first vector.
        vector_b (numpy.ndarray): The second vector.
        
        Returns:
        float: The cosine similarity between the two vectors.
        """
        # Normalize both vectors
        
        total_sum=0
        for index,row in enumerate(matrix_a):
            
            vector_a = matrix_a[index]
            vector_b = matrix_b[index]
            
            max_a = np.max(vector_a)
            max_b = np.max(vector_b)
            
            normalized_vector_a=vector_a
            normalized_vector_b=vector_b
            
            if max_a != 0.0: normalized_vector_a = vector_a / np.linalg.norm(vector_a)
                
            if max_b != 0.0: normalized_vector_b = vector_b / np.linalg.norm(vector_b)
                
            # Calculate the dot product
            s = np.dot(normalized_vector_a, normalized_vector_b)
            total_sum += s
            
        # enqueue cosine similarity
        # self.similarity_queue.append(total_sum / matrix_a.shape[0])
        return total_sum / matrix_a.shape[0]
    
def make():
    return map_2d()

import os
import numpy as np
import json
import store_img_as_json
import random
import cv2

os.system('cls')

class neuronal_network_effort:
    
    gamma = 0.3
    grid=np.zeros((1026,1026))
    
    def __init__(self) -> None:
        self.get_qtable_dataset()
        
    def get_qtable_dataset(self , show_img=0):
                
        dataset = store_img_as_json.store_img_as_json()
        
        for index,sample in enumerate(dataset.load_dataset_json()):
            
            new_qtable = np.zeros( (1026 , 1026 , len( self.actions() * 2 ) )  )
        
            print("data loaded")
            
            x = int(1026/2)
            y =  int(1026/2)
            
            pointer = { "row": y , "column": x }
            
            self.grid = np.zeros((1026,1026))
            self.t = 1026*1026
            self.count = 0
                
            # inital state
            state = {
                    
                    "last_pointer" :{ 'row':0 , 'column':0 },
                    "pointer" :pointer,
                    "qtable":  new_qtable,
            }
            
            print(f'\033[1;32m 0% \033[0m ')

            new_qtable = self.I0( state=state , sample=np.zeros((1026,1026)) , show_img=show_img )
            
            self.store_qtable(qtable=new_qtable , number=index )

        return
    
    def store_qtable( self , qtable:list ,number:int ): 
        
        path = f'./dataset_qtable/qtable{number}.json'
        
        len_actions = len(self.actions())
        
        save_table = []
        for i in range( len(qtable) ):
            c = " "
            for j in range( len(qtable[i]) ):
                
                cell_:cell = self.access_to_target( pointer={ 'row': i , 'column': j } , qtable=qtable )
                
                for i in range(len_actions):
                    
                    action = f'action{i}'
                    c += f'{ str( cell_.__dict__[action] ) } ' # suround each number with white spaces
                
                c +="|" # split each action by a |
            
            save_table.append(c)
        
        with open( path , 'w') as file:
		
        	# Serialize the data to JSON and write it to the file
            json.dump(save_table, file, indent=4)
		
            file.close()
        
        pass
    
    def load_dataset_qtable( self ):
        
        # ----------------------------------------initialize dataset_qtable as qtable0
        path = f'./dataset_qtable/'
        
        path += f'qtable{0}'
        
        reader = open( path,'r')
        
        json_data = reader.read()
        
        reader.close()
        
        data = json.loads(json_data)
        data = self.tokenize_qtable(data=data)
        
        dataset_qtable = np.array(data)
        # ----------------------------------------
        
        for i in range(os.listdir( path=path ) - 1 ):
        
            path += f'qtable{i}'
        
            reader = open( path,'r')
            
            json_data = reader.read()
            
            reader.close()
            
            data = json.loads(json_data)
            data = self.tokenize_qtable(data=data)
            
            np.concatenate( ( dataset_qtable , data ) , 0 )
        
        return dataset_qtable
    
    def tokenize_qtable(self , data ):
        
        qtable=[]
        
        for item in data:
            
            row = []
            item = str(item).split('|') # split by actions
            
            for action in item:
            
                s=  str(action).split(" ")[ 1 : -1 ] # remove the first and last blanck space
                row = [ int(i) for i in s ] # peak each action string and convert
                
            qtable.append(row)
            
        return qtable
    
    def init_qtable( self ,len_row , len_column ):
        
        qtable = []
        len_actions =len( self.actions() )
        for i in range(len_row):
            
            row = []
            for j in range(len_column):
                row.append( cell( len_actions=len_actions ) ) # initialize a matrix of len_row size x len_column size

            qtable.append(row)
            
        return qtable
    
    def reward_function( self, state , action , sample ):
        
        '''
        ### This function models problem's constrains
        '''
        
        pointer = state['pointer']
        
        if pointer['row']  + action[1] < 0 or pointer['row']  + action[1] >= 1026:
            return float('-inf')
        
        if  pointer['column']  + action[2] < 0 or pointer['column'] + action[2] >= 1026:
            return float('-inf')
                
        # if sample[ pointer['row'] + action[1] , pointer['column'] + action[2] ] == 255:
        #     return float('-inf')
            
        return 10
    
    def actions(self):
        
        import random
        
        action_table = []
        uper_bound = 4
        lower_bound = -4
        radio = 0
        
        for i in range(lower_bound , uper_bound):
            for j in range(lower_bound , uper_bound):
                action_table.append( ( True , i + radio , j + radio ) )
        
        x= random.randint( -300 , 300 )
        y= random.randint( -300 , 300 )
        action_table.append( ( 1 , x , y  ) )
        # action_table.append( ( 0 , x , y  ) )

        return action_table
    
    def best_action( self, state , sample ): # best probability action
        
        actions = self.actions()
        
        k = 1
        rw_best_action = [0]
        best_action = [] # best action could be a list of best actions where each action has the same reward
        
        for index,action in enumerate(actions):
            
            rw_action = k ** self.reward_function( state=state , action=action , sample=sample)
            
            sum_rw = 1
            
            for index2,action_avg in enumerate(actions):
                
                if index == index2: 
                    continue
                
                sum_rw += k ** self.reward_function(state , action_avg , sample)
            
            rw_action /= sum_rw
    
            if rw_best_action[0] < rw_action:
                
                rw_best_action = [rw_action]
                best_action = [action]
                
            elif rw_best_action[0] == rw_action:
                
                best_action.append(action)
                rw_best_action.append(rw_action)
        
        random.shuffle(best_action)
        
        return  best_action
    
    def I0( self , state , sample , show_img=0 ):
        
        for index,new_action in enumerate(self.best_action( state=state , sample=sample )):
            
            inmediate_rw = self.reward_function( state=state, action=new_action , sample=sample )
            
            if inmediate_rw == float('-inf'):
                continue
            
            new_state , sample = self.apply_action( state=state, action=new_action , sample=sample ) # apply action just at cycle start
            
            stack = [  
            {
                'state':new_state , 
                'sample':sample ,
                'actual_action':new_action ,
                'action_num':index ,
                'call_number': 0 ,
                'children_return': [],
                'return_to':0,
                'inmediate_rw': inmediate_rw ,
                'a_n':1 ,    
            } 
        ]

            stack = self.qfunction( stack=stack , show_img=show_img ) # action at the new builded state
            
            state_max=stack[0]['children_return'][0][0]
            max_rw=stack[0]['children_return'][0][1]
            for item in stack[0]['children_return']: # update max passing by every child return
                
                if item[1] > max_rw:
                    
                    state_max = item[0]
                    max_rw = item[1]
                
            # reward of the state with actual action applied Q(s,a) <- (1 - a_n) * Q_(n-1)(s,a) a_n(r + y * max( Q(s',a') ))
            state_rw =  inmediate_rw + self.gamma * max_rw
            
            target:cell = self.access_to_target( state_max['qtable'] , state_max['pointer'] )
            
            target.__dict__[ f'action{index}' ] = state_rw
            state= state_max
            sample =stack[0]['sample']
                
        return state['qtable']
    
    def apply_action(self , state , action , sample , show_img=0):
        
        # apply action
        state['last_pointer']['row'] = state['pointer']['row']
        state['last_pointer']['column'] = state['pointer']['column']
        state['pointer']['row'] +=action[1]         
        state['pointer']['column'] += action[2] 
        
        # print( state['pointer']['row'] , state['pointer']['column'] )
        i = state['pointer']['row']
        j = state['pointer']['column']
        
        if action[0]:
            sample[i,j] = 255
        else:
            sample[i,j] = 0
        
        if show_img and self.count - self.last_count == 10000:
            
            self.last_count = self.count
            self.show_img(sample=sample)
        
        return state , sample
    
    def show_img(self , sample):
        
        normalized_color_array = np.clip(sample, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(normalized_color_array, cv2.COLOR_GRAY2BGR)
        cv2.imshow('kk',image)
        cv2.waitKey(0)

        return
    
    def qfunction( self , stack , show_img=0 ): 
        
        pointer = 0
        len_actions = len( self.actions() )
        while len(stack) != 0:
            
            if self.last_percent < int((self.count / self.t) * 100):
                os.system('cls')
                self.last_percent = int((self.count / self.t) * 100 )
                print(f'\033[1;32m {int((self.count / self.t)* 100 ) }% \033[0m ')
            
            # print( len(stack) , pointer )
            if len(stack) <= pointer: break
            
            call = stack[pointer]
            sample = call['sample']
            state = call['state']
            call_number = call['call_number']
            
            for index,new_action in enumerate(self.best_action( state=state , sample=sample )):
            
                inmediate_rw = self.reward_function( state=state , action=new_action , sample=sample )
                
                if inmediate_rw == float('-inf'):
                
                    stack[call_number]['children_return'].append( [ state , float('-inf') ] )
                    continue
                
                new_state , sample = self.apply_action( state=state, action=new_action , sample=sample , show_img=show_img ) # apply action just at cycle start
                
                state['qtable'][ state['pointer']['row'] , state['pointer']['column'] , index + len_actions ] += 1
                visits = self.access_to_target( new_state['qtable'] , new_state['pointer'] , index=index + len_actions  )
                a_n = self.a_n( visits ) # number of visits at this state
                
                # print( "visit=" , target.__dict__[ visit_action ] , "a_n=" , a_n)
                
                if a_n == 0:
                    stack[call_number]['children_return'].append( [ state ,  update_rw ] )
                    continue
                
                # reward at actual state applying actual action
                update_rw = self.access_to_target( new_state['qtable'] , new_state['pointer'] , index=index )
                
                new_call = {
                    
                    'state':new_state,
                    "sample": sample,
                    'call_number': len(stack),             
                    'inmediate_rw': inmediate_rw,
                    'update_rw': update_rw,
                    'children_return':[],
                    'return_to':call_number,
                    'a_n':a_n,
                    'action':index,
                }
                
                stack.append(new_call)
            
            pointer += 1
        
        children_return = self.fill_return_list(stack=stack)        
        
        return children_return

    def fill_return_list(self , stack ):
        
        while len(stack) != 1:
            
            call = stack[-1]
            stack.pop()
            
            a_n = call['a_n']
            update_rw = call['update_rw']
            inmediate_rw = call['inmediate_rw']
            action = call['action']
            state = call['state']
            
            state_max=call['children_return'][0][0]
            max_rw=call['children_return'][0][1]
            for item in call['children_return']: # update max passing by every child return
                
                if item[1] > max_rw:
                    
                    state_max = item[0]
                    max_rw = item[1]
                
            # reward of the state with actual action applied Q(s,a) <- (1 - a_n) * Q_(n-1)(s,a) a_n(r + y * max( Q(s',a') ))
            state_rw = ( 1 - a_n) * update_rw + a_n * ( inmediate_rw + self.gamma * max_rw)
            
            if state_rw > update_rw : # reward might be negative , so we do not update qtable
                state['qtable'][ state['pointer']['row'] , state['pointer']['column'] , action ] = state_rw
                
            stack[ call['return_to'] ]['children_return'].append( [ state_max , state_rw ] )
        
        return stack
    
    t = 1026 * 1026
    count = 0
    last_count = 0
    last_percent = 0
    def access_to_target( self, qtable , pointer , index=0 ):
        
        if not self.grid[ pointer['row'] , pointer['column'] ]:
            self.count += 1

            # print(   int(self.count / self.t) )
            
            self.grid[ pointer['row'] , pointer['column'] ] = 1
        
        return qtable[ pointer['row'] , pointer['column'] , index ]
    
    def a_n(self, num_visit ):
        
        if num_visit > 20:
            return 0.0
        
        return 1 / ( 1 + num_visit   )

class cell:
        
    def __init__(self, len_actions ):
        
        for i in range(len_actions):
            self.__dict__[f'action{i}'] = 0
            self.__dict__[f'visit_action{i}'] = 0
            
        pass

nw = neuronal_network_effort()

import numpy as np
import os
import cv2
import math



def generate_dataset(number=3):
    
    def matrix_cosine_similarity( matrix_a , matrix_b ):
        
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
        
        return total_sum / matrix_a.shape[0]
    
    print('number:',number)
    path_to_patch = './patches'
    
    if not f'x{number}' in os.listdir('./data'):
        os.mkdir(f'./data/x{number}')
        os.mkdir(f'./data/y{number}')
    
    patch = os.listdir(path=path_to_patch)[number]
    
    img_patch = cv2.imread( path_to_patch +"/" + patch , cv2.IMREAD_GRAYSCALE ) # patch to compare
    img_patch = np.where( img_patch > 128 , 1.0 , 0.0 )
    
    patch_white = './dataset_black_white/train'
    target = os.listdir(path=patch_white)[0]
    img_target = cv2.imread( patch_white + "/" + target , cv2.IMREAD_GRAYSCALE ) # target to crop
    img_target = np.where( img_target > 128 , 1.0 , 0.0 )
    
    sample = np.zeros((100,100), dtype='uint8')
    count = -1
    desplacement = 100
    for patch_row in range(int(img_target.shape[1] / desplacement)):
        for patch_column in range(int(img_target.shape[0] / desplacement)):
                count += 1
                print( 'number created: ', count , 'number left: ', int(img_target.shape[0] / desplacement) * int(img_target.shape[1] / desplacement) - count )
                for i in range( 100 ): # extract patch
                    
                    k = patch_row * desplacement + i
                    if k >= img_target.shape[0]: break
                    
                    for j in range( 100 ):
                        l= patch_column* desplacement + j
                    
                        if l >= img_target.shape[0]: break
                        sample[ j , i  ] = img_target[l,k]
                
                cv2.imwrite( f'./data/x{number}/{count}.jpg' , sample * 255.0 )
                similarity = 0
                
                for i1 in range(sample.shape[1]): # apply actions over a sample verifing that similarity increases
                    for j1 in range(sample.shape[0]):
                        last_value = sample[j1,i1]
                        sample[j1,i1] = 1.0
                        sim =matrix_cosine_similarity( sample , img_patch )
                        if  sim > similarity:
                            similarity = sim
                        else:
                            sample[j1,i1] = last_value

                print(f'iteraction { i1 + j1 }...')
                cv2.imwrite( f'./data/y{number}/{count}.jpg' , sample * 255.0 )
                
                
def train_model(use = 0, version=0 , img:np.array=None):
    
    import keras.models as m
    
    def use_model(version , img_target ):
        
        model = m.load_model(f'./model_v{version}')
        
        
        _input_ = img_target
        _input_ = np.expand_dims( _input_ , 0 )
        _input_ = np.expand_dims( _input_ , -1 )
        
        output = np.clip(  model(_input_,training=False) , 0.0 , 1.0 )
        
        output = np.squeeze(output)
        
        output = np.reshape(output , (100,100))
        
        return output
        
    if use: return use_model( version=version , img_target=img )
            
    def load_data(number):
        print('loading data...')
        data_x = f'./data/x{number}'
        data_y = f'./data/y{number}'

        data_input = []
        for sample in os.listdir(data_x):
            img = cv2.imread( data_x +'/'+ sample , cv2.IMREAD_GRAYSCALE )
            img = np.where(img > 128 , 1.0, 0.0 )
            data_input.append( img )
        
        data_output = []
        for sample in os.listdir(data_y):
            img = cv2.imread( data_y +'/'+ sample , cv2.IMREAD_GRAYSCALE )
            img = np.where(img > 128 , 1.0, 0.0 )
            data_output.append( img.flatten() )
    
        if len(data_input) != len(data_output):
            
            if len(data_input) > len(data_output):
                diff = len(data_input) - len(data_output)
                for i in range(diff):
                    data_input.pop()
            
            else:
                
                diff = len(data_output) - len(data_input)
                for i in range(diff):
                    data_input.pop()
                
    
        per_cent = .8
        x_train = data_input[: int(len(data_input) * per_cent)  ]     
        y_train = data_output[: int(len(data_output) * per_cent)  ]     
        
        
        x_val = data_input[int(len(data_input) * per_cent): ]     
        y_val = data_output[int(len(data_output) * per_cent):]     
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        return  x_train ,  y_train ,  np.array(x_val) , np.array(y_val) 
    
    import keras
    from keras import layers
    import  tensorflow as tf
    
    def create_model(input_data= (100,100) ):
        
        # Convolutions on the frames on the screen
        input_layer=layers.Input( shape= ( input_data[0] , input_data[1] , 1 ) )
        layer=layers.Dropout(rate=.1) (input_layer)
        layer = layers.Conv2D(32 , 8 , activation='linear')(layer)
        
        
        layer=layers.Flatten()(layer)
        layer=layers.Dense(10, activation="linear" )(layer)
        layer=layers.Dense(10, activation="linear" )(layer)

        
        output=layers.Dense( input_data[0] * input_data[1] , activation="linear")(layer)
        
        model = keras.Model(inputs=input_layer, outputs=output )
        
        return model
    
    def train(number ):
        
        model = create_model()
        model.summary()
        
        optimizer = keras.optimizers.Adam(learning_rate=.0001 )
        loss_function = keras.losses.mse
        model.compile( optimizer=optimizer , loss = loss_function , metrics=['accuracy'] )
        
        x_train , y_train , x_val , y_val = load_data(number=number)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        
        indices = np.arange(len(x_train))
            
        import matplotlib.pyplot as plt
        
        history = { 'loss':[] , 'val_loss':[] }
        epochs = 0
        
        epsilon_stop = 5/1000
        while True:
            
            i = 0
            np.random.shuffle(indices) # shuffle train dataset 
            x_train = x_train[indices] 
            y_train = y_train[indices]
            
            mean_precision = [ ]
            while i < len( x_train ): # adjust weight of the model
                
                    
                with tf.GradientTape() as tape:
                    x = x_train[i]
                    x = np.expand_dims(x,0)
                    x = np.expand_dims(x,-1)
                    y = model(x , training = True )
                    
                    cv2.imwrite( './prediction.jpg' , np.resize( y , (100,100) ).astype('uint8') * 255.0 ) # write image
                    
                    cv2.imwrite( 
                                './original.jpg' ,
                                np.resize( y_train[i] , (100,100))  * 255.0 ) # write image
                    
                    loss = loss_function(  y , y_train[i] )
                    mean_precision.append( int(math.log10( 1/ loss.numpy()[0] )) )

                    if len(mean_precision) == 100: mean_precision.pop(0)
                    
                h =  int( np.array(mean_precision).mean() + 1 ) * 10 
                
                if (i + 1) % h == 0 :
                    
                    grads = tape.gradient( loss , model.trainable_variables )
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    
                    os.system('cls')
                    print('presicion: ', h/10 )
                    print(f'epoch: {epochs}')
                    print( f'sample_train: {i}/{len(x_train)}' )
                    print( 'loss: ',loss.numpy()[0] )
                    m.save_model(model=model , filepath=f'./model_v{number}')
                
                history['loss'].append(loss.numpy()[0])
                i += 1
            
            i = 0
            val_loss = []
            while i < len( x_val): # validate training
                
                
                with tf.GradientTape() as tape:
                    x = x_val[i]
                    x = np.expand_dims(x,0)
                    x = np.expand_dims(x,-1)
                    y = model(x ,training=False)
                    cv2.imwrite( './prediction.jpg' , np.resize( y , (100,100) ).astype('uint8') * 255.0 ) # write image
                    cv2.imwrite( './original.jpg' , np.resize( y_val[i] , (100,100) ).astype('uint8') * 255.0 ) # write image
                    
                    loss = loss_function(  y , y_val[i] )
                    val_loss.append(loss)
                    
                if i % 10==0 :
                    os.system('cls')
                    print(f'epoch: {epochs}')
                    print( f'sample_val: {i}/{len(x_val)}' )
                    print( 'loss: ',loss.numpy()[0] )
                    
                history['loss'].append(loss.numpy()[0])
                i += 1
            
            val_loss = np.array(val_loss)
            if val_loss.mean(0) <= epsilon_stop: break
        
    
        # plotting results
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./report.png')
        plt.close()

    for i in range( 12 , num ):
        train(number=i)
    
def start_generation(num):
    
    generate_dataset(num)
    

def paralell_generation_dataset():
    import threading

    patches = os.listdir('./patches')
    threads = []
    for i in range(num,len(patches)):
        print('thread:',i)
        num += 1
        ti = threading.Thread(target=start_generation,args=([num]))
        ti.start()
        threads.append(ti)

    for i in threads:
        i.join()
        

num = 100

train_model()
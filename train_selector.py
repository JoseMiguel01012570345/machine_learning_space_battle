import keras
import keras.layers as layers
import numpy as np
import os
import cv2
from signals import *
import tensorflow as tf


os.system('cls')

def create_model_selector( input_data=5 ):
        
    
    # First convolutional layer
    input_layer=layers.Input( shape= ( input_data , ) )
    
    # Dense layers for classification
    x = layers.Dense(64, activation='linear')(input_layer)
    x = x / tf.abs(x)
    x = x / tf.reduce_max(x)
    
    x = layers.Dense(32, activation='linear')(x)
    
    x = x / tf.abs(x)
    x = x / tf.reduce_max(x)
    
    output = layers.Dense( 12, activation='linear')(x) # For 12 classes

    x = x / tf.abs(x)
    x = x / tf.reduce_max(x)

    return keras.Model(inputs= input_layer , outputs=output )

def load_data(number):
        
        print('loading data...')
        data_x = f'./data/x{number}'
        

        data_input = []
        data_output = []
        for sample in os.listdir(data_x):
            img = cv2.imread( data_x +'/'+ sample , cv2.IMREAD_GRAYSCALE )
            img = np.where(img > 128 , 1.0, 0.0 )
            signal = process_sector(img) 
            
            x = np.array( [ signal.Centroid , signal.Average , signal.Contrast ,  signal.Flatness , signal.OnesAverage ] )
            
            x = np.where( ~np.isnan(x) , x , 0 )
            x = x / np.max(x)
            data_input.append( x )
            
            data_output.append( number )
            
        
        per_cent = .7
        x_train = data_input[: int(len(data_input) * per_cent)  ]     
        y_train = data_output[: int(len(data_output) * per_cent)  ]
        
        
        x_val = data_input[int(len(data_input) * per_cent): ]     
        y_val = data_output[int(len(data_output) * per_cent):]     
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        
        print( 'max_dataset: ', np.max( np.array(x_train)) ,'min_dataset: ', np.min( np.array(x_train)) )
        
        return  x_train ,  y_train ,  np.array(x_val) , np.array(y_val)
num_patches = 100

def process_sector(matrix):
    samples = matrix2canonics_coefs(matrix)
    return get_signal_features(samples)

def train_selector():
    
    x_train ,  y_train ,  x_val , y_val = [] ,[] , [] , []
    
    model = create_model_selector()
    model.summary()
    epsilon_rate = .0001
    
    import matplotlib.pyplot as plt
    
    while True:
    
        for i in range(12):
            x_train ,  y_train ,  x_val , y_val = load_data( number=i )
            
            optimizer = keras.optimizers.Adam(learning_rate=epsilon_rate )
            model.compile( optimizer=optimizer , loss = 'mae' , metrics=['accuracy'] )
                
            epochs = 100
            history = model.fit( x=x_train , y=y_train , validation_data=( x_val , y_val) , epochs=epochs , shuffle=True , batch_size=32 )
            history = history.history
            keras.models.save_model(model=model , filepath=f'./model_selector')

            epsilon_rate += .00001
            
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.plot(history['accuracy'], label='accuracy')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./report.png')
            plt.close()
        
        x_train ,  y_train ,  x_val , y_val = [] ,[] , [] , []

        for i in range(12):
            x_train0 ,  y_train0 ,  x_val0 , y_val0 = load_data( number=i )
            
            x_train.extend(x_train0)
            y_train.extend(y_train0)
            x_val.extend(x_val0)
            y_val.extend(y_val0)
            
            epochs = 100
            history = model.fit( x=np.array(x_train) , y=np.array(y_train) , validation_data=( np.array(x_val) , np.array(y_val)) , epochs=epochs , shuffle=True , batch_size=32 )
            history = history.history
            keras.models.save_model(model=model , filepath=f'./model_selector')

            epsilon_rate += .00001
            import matplotlib.pyplot as plt
            
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.plot(history['accuracy'], label='accuracy')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('./report.png')
            plt.close()
            
        x_train ,  y_train ,  x_val , y_val = [] ,[] , [] , []
        
def use_model( x:np.array= None  , model_list:list = [] ):
    
    best_similarity = 1e305
    best_img = np.array([])
    
    for v in range(num_patches):
        
        z = x 
        model = model_list[v]
        z = np.expand_dims(z , 0)
        z = np.expand_dims(z , -1)
        y = model( z , training=False )
        y = np.squeeze(y)
        y = y / np.max(y)
        y = np.clip( y , 0.0 , 1.0 )
        y = np.where( y >= (np.max(y) + np.min(y))/2 , 1.0 , 0.0 )
        y = np.reshape( y , (100,100) )
        similarity = np.mean( keras.losses.mae( y , x ).numpy())
        
        if similarity < best_similarity:
            best_similarity = similarity
            best_img = y
        
    return best_img
         

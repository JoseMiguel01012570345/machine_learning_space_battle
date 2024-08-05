from keras import layers
from keras import Input
from keras import models
from keras import optimizers
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
import json
import os
os.system("cls")

def load_data( number ):
    
    train_dir = f'./dataset/train/dataset{number}/'
    val_dir=f'./dataset/val/dataset{number}/'
    
    aux = []
    print(f"\033[1;32m loading training data {train_dir} \033[0m")
    for json_data in os.listdir(train_dir):
        
        file= open( train_dir + json_data , 'r' ).read()
        tok = tokenize_data(json_data=file)
        aux.append( tok )
        
        pass
        
    train_data_x = np.stack( aux, axis=0)
    
    file= open( val_dir + f'image{ number }.json' , 'r' ).read()
    val = tokenize_data(json_data=file)
    val = val.flatten()
    
    train_data_y = np.array([ val for i in range(len(train_data_x)) ])
    
    # 80% of training
    train = { "train_x": train_data_x[ 0 : int( len(train_data_x) * .8 ) ] , "train_y": train_data_y[ 0 : int( len(train_data_y) * .8 ) ] }
    
    # 20% of validation
    val = { "val_x": train_data_x[ int( len(train_data_x) * .8 ) :  ] , "val_y": train_data_y[ int( len(train_data_y) * .8 ) : ] }

    return train , val

def tokenize_data(json_data):
    
    data = json.loads(json_data)
    
    image = np.ones((800,600))
    
    for i,row in enumerate(data):
        
        token_row = data[row].split(' ')[1:-1]
        for item in token_row:
            
            image[int(item),i] = 0
    
    return image

def network():
    
    # Encoder
    model = models.Sequential([
    
    layers.Input( shape=( 800 , 600 , 1 ) ),
    layers.MaxPooling2D((25, 20) ),
    
    # First convolutional layer
    layers.Conv2D(16, (3, 3), padding='same' ,  activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    layers.Conv2D(32, (3, 3), padding='same' , activation='relu'),
    layers.MaxPooling2D((2, 3)),
    
    # Third convolutional layer
    layers.Conv2D(32, (3, 3), padding='same' , activation='relu'),
    
    # Fourth convolutional layer
    layers.Conv2D(32, (3, 3), padding='same' ,activation='relu'),
    
    layers.Dense(512, activation='relu'), 
    layers.Dense(12000, activation='relu'),    
    # Flattening the 3D output to 1D before feeding it into the dense layer
    layers.Flatten(),
    
    # Dense layers for classification    
    ])
    
    return model

def engine():

    print("\033[1;32m building model \033[0m")
    
    # Create a new model for classification
    auto_encoder = network()
    
    # Summary of the model
    auto_encoder.summary()
    
    print("\033[1;32m compiling model \033[0m")
    
    auto_encoder.compile(optimizer= 'adam',
                            loss='mse',
                            metrics=['accuracy'])
    
    epochs = 4
    history = None
    
    breaking = 1
    for i in range( len(os.listdir('./dataset/train/') )):
        
        if i != breaking: continue
        
        train_data_x , val_data_y = load_data(i)
        
        print(f"\033[1;32m fitting model {i} \033[0m")
        
        history = auto_encoder.fit(train_data_x['train_x'] , train_data_x['train_y'] , batch_size=1, epochs=epochs,verbose=1 , validation_data=(val_data_y['val_x'] , val_data_y['val_y'] ) , use_multiprocessing=True )
        
    
    auto_encoder.save('autoencoder_shape=1026x1026.h5')  # Creates a HDF5 file 'my_model.h5'
    
    return history , epochs

def visualize_training( autoencoder_train , epochs ):

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

model , epochs = engine()
visualize_training( model , epochs )

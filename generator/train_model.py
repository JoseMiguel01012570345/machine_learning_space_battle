import keras
from keras import layers
from keras import Input
from keras import Model
import numpy as np
import cv2
import matplotlib as plt
from matplotlib import pyplot as plt

import os
os.system("cls")

def dataset_train_validator():
    
    train_dir = './img_map_dataset/'
    
    aux = []
    print("\033[1;32m loading training data \033[0m")
    for img in range( len(os.listdir(train_dir))):
        
        image = cv2.imread(train_dir + f"/map{img}.jpg", cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
        
        image = image.astype(np.float32) / 255
        aux.append(image)
    
    data = np.stack( aux, axis=0)

    num_train = int(len(os.listdir(train_dir)) * 0.8)
    
    train_data = data[ 0 : num_train ]
    val_data = data[ num_train: ]
    
    
    return train_data , val_data

def autoencoder(input_img):
    
    # Encoder
    x = layers.MaxPooling2D(pool_size=(171, 171))(input_img) 
    x = layers.Conv2D(8 , (3,3) , padding='same', activation='relu')(x) 
    x = layers.Dense(32 , activation='relu')(x) 
    
    x = layers.Conv2D(4 , (3,3) , padding='same', activation='relu')(x) 
    x = layers.Dense(16 , activation='relu')(x) 
    
    x = layers.MaxPooling2D(pool_size=(3, 3 ))(x) 
    x = layers.Conv2D(2 , (3,3) , padding='same', activation='relu')(x) 
    encode = layers.Dense(8 , activation='relu')(x) 
    
    #decoder
    x = layers.UpSampling2D((3,3))(encode) 
    x = layers.Conv2D(2 , (3,3) , padding='same' , activation='relu' )(x)
    x = layers.Dense(8, activation='relu')(x) 
    
    x = layers.Conv2D(4 , (2,2) , padding='same' , activation='relu' )(x)
    x = layers.Dense(16, activation='relu')(x) 
    
    x = layers.UpSampling2D((171,171))(x) 
    decoded = layers.Dense(1, activation='relu')(x) 
    
    return decoded    

def engine():
        
    print("\033[1;32m building model \033[0m")
    
    train_data , val_data = dataset_train_validator()
    
    # Input shape: (batch_size, 28, 28, 1)
    input_img = Input(shape=( 1026, 1026 , 1 ))

    # Create a new model for classification
    auto_encoder = Model(inputs=input_img, outputs=autoencoder(input_img))
    
    # Summary of the model
    auto_encoder.summary()

    # Compile the model
    
    print("\033[1;32m compiling model \033[0m")
    
    auto_encoder.compile(optimizer= 'Adam',
                             loss='mean_squared_error',
                             metrics=['accuracy'])

    epochs = 5
    
    print("\033[1;32m fitting model \033[0m")
    
    autoencoder_train = auto_encoder.fit(train_data , train_data , batch_size=1,epochs=epochs,verbose=1,validation_data=(val_data,val_data))
    
    auto_encoder.save('autoencoder_shape_1026.h5')  # Creates a HDF5 file 'my_model.h5'
    
    return autoencoder_train , epochs

def visualize_training( autoencoder_train  ,epochs ):

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
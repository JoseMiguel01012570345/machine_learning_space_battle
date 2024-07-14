import keras
import tensorflow as tf
from keras import layers
from keras import Input
from keras import Model
import numpy as np
import cv2
import matplotlib as plt

import os
os.system("cls")

def autoencoder(input_img):
    
    # Encoder
    x = layers.MaxPooling3D(pool_size=(38, 1, 1 ))(input_img) #5 x 5 x 64
    x = layers.Conv3D(8 , (3,3,3) , padding='same', activation='relu')(x) #30 x 30 x 32
    x = layers.Dense(32 , activation='relu')(x) #30 x 30 x 32
    
    x = layers.MaxPooling3D(pool_size=(2, 3 , 3 ))(x) #15 x 15 x 32
    x = layers.Conv3D(4 , (3,3,3) , padding='same', activation='relu')(x) #30 x 30 x 32
    x = layers.Dense(16 , activation='relu')(x) #15 x 15 x 64
    
    
    x = layers.MaxPooling3D(pool_size=(1, 3 , 3 ))(x) #15 x 15 x 32
    x = layers.Conv3D(2 , (3,3,3) , padding='same', activation='relu')(x) #30 x 30 x 32
    encode = layers.Dense(8 , activation='relu')(x) #15 x 15 x 64
    
    #decoder
    x = layers.UpSampling3D((1,3,3))(encode) # 15 x 15 x 128
    x = layers.Conv3D(2 , (3,3,3) , padding='same' , activation='relu' )(x)
    x = layers.Dense(8, activation='relu')(x) # 15 x 15 x 64
    x = layers.UpSampling3D((2,3,3))(x) # 30 x 30 x 64
    
    x = layers.Conv3D(4 , (2,2,2) , padding='same' , activation='relu' )(x)
    x = layers.Dense(16, activation='relu')(x) # 15 x 15 x 64
    x = layers.UpSampling3D((38,1,1))(x) # 30 x 30 x 64
    decoded = layers.Dense(1, activation='relu')(x) # 30 x 30 x 1
    
    return decoded    

def dataset_train_validator():
    
    train_dir = './img_map_dataset/'
    
    aux = []
    print("\033[1;32m loading training data \033[0m")
    for img in range( len(os.listdir(train_dir))):
        
        image = cv2.imread(train_dir + f"/map{img}.jpg", cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
        
        image = image.astype(np.float32) / 255
        aux.append(image.reshape( 1444,27,27 ) )
    
    data = np.stack( aux, axis=0)

    num_train = int(len(os.listdir(train_dir)) * 0.8)
    
    train_data = data[ 0 : num_train ]
    val_data = data[ num_train: ]
    
    
    return train_data , val_data

def engine():
    
    hyper_params = []
    
    print("\033[1;32m building model \033[0m")
    
    train_data , val_data = dataset_train_validator()
    
    # Input shape: (batch_size, 28, 28, 1)
    input_img = Input(shape=( 1444 , 27, 27, 1 ))

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
    
    auto_encoder.save('autoencoder.h5')  # Creates a HDF5 file 'my_model.h5'
    
    pass

def visualize_training( autoencoder_train ):

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def use_model():
    
    test_dir = './img_map_dataset/'
    
    for img in range( len(os.listdir(test_dir))):
    
        test = cv2.imread('./dataset_balck_white/train/0.jpg').reshape(1444,27,27)
        model = keras.load('autoencoder.h5')
    
    model.predict(test)
    
    pass
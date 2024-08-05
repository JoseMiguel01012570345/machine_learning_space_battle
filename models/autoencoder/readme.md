_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 1026, 1026, 1)]   0

 max_pooling2d (MaxPooling2  (None, 171, 171, 1)       0
 D)

 conv2d (Conv2D)             (None, 171, 171, 32)      320       

 dense (Dense)               (None, 171, 171, 32)      1056      

 max_pooling2d_1 (MaxPoolin  (None, 57, 57, 32)        0
 g2D)

 conv2d_1 (Conv2D)           (None, 57, 57, 16)        4624      

 dense_1 (Dense)             (None, 57, 57, 16)        272       

 max_pooling2d_2 (MaxPoolin  (None, 19, 19, 16)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 19, 19, 8)         1160      

 dense_2 (Dense)             (None, 19, 19, 8)         72        

 up_sampling2d (UpSampling2  (None, 57, 57, 8)         0
 D)

 conv2d_3 (Conv2D)           (None, 57, 57, 4)         292       

 dense_3 (Dense)             (None, 57, 57, 4)         20        

 up_sampling2d_1 (UpSamplin  (None, 171, 171, 4)       0
 g2D)

 conv2d_4 (Conv2D)           (None, 171, 171, 8)       136

 dense_4 (Dense)             (None, 171, 171, 8)       72

 up_sampling2d_2 (UpSamplin  (None, 1026, 1026, 8)     0
 g2D)

 dense_5 (Dense)             (None, 1026, 1026, 1)     9

=================================================================
##
Total params: 8033 (31.38 KB)
##
Trainable params: 8033 (31.38 KB)
##
Non-trainable params: 0 (0.00 Byte)
##
### Loss: 0.1148 <-> accuracy: 0.7667
_________________________________________________________________

- $\left(\frac{W - P}{S} + 1, \frac{H - Q}{T} + 1, D\right)$
- $\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$
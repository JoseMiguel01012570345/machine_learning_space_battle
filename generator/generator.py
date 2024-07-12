from keras import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Input
from keras import Model

import os
os.system("cls")

def dataset_train_validator():
    
    train_dir = '../dataset/train'
    validation_dir = '../dataset/validation'
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
#     len_train = len(os.listdir(train_dir + "/earth/" ) )
#     len_val = len(os.listdir(validation_dir + "/earth/" ) )
    

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='input')

    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='input')

    return train_generator , validation_generator

def classfier( y_true, y_pred ):
        os.system("cls")
        print(y_pred)
        '''
        loss: Loss function. May be a string : name of loss function , or
        a tf.keras.losses.Loss instance. See tf.keras.losses. A loss function is any callable with the signature loss = fn(y_true, y_pred), 
        where y_true are the ground truth values, and y_pred are the model's predictions. 
        y_true should have shape (batch_size, d0, .. dN) (except in the case of sparse loss functions such as sparse categorical crossentropy which 
        expects integer arrays of shape (batch_size, d0, .. dN-1)). y_pred should have shape (batch_size, d0, .. dN). The loss function should return 
        a float tensor. If a custom Loss instance is used and reduction is set to None, return value has shape (batch_size, d0, .. dN-1) i.e. per-sample
        or per-timestep loss values; otherwise, it is a scalar. If the model has multiple outputs, you can use a different loss on each output by passing 
        a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses, unless loss_weights is specified.

        Configures the model for training.

        Example:

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
        Args
        optimizer: String : name of optimizer or optimizer instance. See
        tf.keras.optimizers.

        metrics : List of metrics to be evaluated by the model during
        training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. 
        Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for 
        different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You
        can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy',
        ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, 
        tf.keras.metrics.SparseCategoricalAccuracy based on the shapes of the targets and of the model output. We do a similar conversion for the strings 'crossentropy' and 
        'ce' as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the 
        weighted_metrics argument instead.

        loss_weights : Optional list or dictionary specifying scalar
        coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted
        sum of all individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is 
        expected to map output names (strings) to scalar coefficients.

        weighted_metrics : List of metrics to be evaluated and weighted by
        sample_weight or class_weight during training and testing.

        run_eagerly : Bool. If True, this Model's logic will not be
        wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. run_eagerly=True is not supported when using 
        tf.distribute.experimental.ParameterServerStrategy. Defaults to False.

        steps_per_execution : Int or 'auto'. The number of batches to
        run during each tf.function call. If set to "auto", keras will automatically tune steps_per_execution during runtime. Running multiple batches inside a single
        tf.function call can greatly improve performance on TPUs, when used with distributed strategies such as ParameterServerStrategy, or with small models with a 
        large Python overhead. At most, one full epoch will be run each execution. If a number larger than the size of the epoch is passed, the execution will be truncated
        to the size of the epoch. Note that if steps_per_execution is set to N, Callback.on_batch_begin and Callback.on_batch_end methods will only be called every N batches
        (i.e. before/after each tf.function execution). Defaults to 1.

        jit_compile : If True, compile the model training step with XLA.
        XLA is an optimizing compiler for machine learning. jit_compile is not enabled for by default. Note that jit_compile=True may not necessarily work for all models. 
        For more information on supported operations please refer to the XLA documentation. Also refer to known XLA issues for more details.

        pss_evaluation_shards : Integer or 'auto'. Used for
        tf.distribute.ParameterServerStrategy training only. This arg sets the number of shards to split the dataset into, to enable an exact visitation guarantee for evaluation,
        meaning the model will be applied to each dataset element exactly once, even if workers fail. The dataset must be sharded to ensure separate workers do not process the 
        same data. The number of shards should be at least the number of workers for good performance. A value of 'auto' turns on exact evaluation and uses a heuristic for the 
        number of shards based on the number of workers. 0, meaning no visitation guarantee is provided. NOTE: Custom implementations of Model.test_step will be ignored when doing 
        exact evaluation. Defaults to 0.

        **kwargs : Arguments supported for backwards compatibility only.
        
        '''

        return 

def generator():
    
    hyper_params = []
    
    # Input shape: (batch_size, 28, 28, 1)
    input_img = Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(encoded)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Classifier head    
    x = layers.Flatten()(decoded)  # Flatten the output of the decoder
    dense_classifier = Dense( 1026 * 1026 , activation='softmax')(x)  # Add a dense layer for dimensionality reduction

    # Create a new model for classification
    classifier_model = Model(inputs=input_img, outputs=dense_classifier)
        
    
    # Summary of the model
    classifier_model.summary()

    # Compile the model
    classifier_model.compile(optimizer='adam',
                             loss=classfier,
                             metrics=['accuracy'])

    train_generator , validation_generator = dataset_train_validator()
    
    epoch = 10
    
    history = classifier_model.fit(
        
        train_generator,
        epochs=epoch,  # Number of epochs
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
    )

#     test_loss, test_acc = autoencoder.evaluate(test_generator, test_labels)
    
#     print('\nTest accuracy:', test_acc)
    
    pass

generator()
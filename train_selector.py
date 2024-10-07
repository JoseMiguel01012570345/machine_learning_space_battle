import numpy as np
import os
import classifiers_algorithm
import joblib
import json


def create_model_selector( input_data=5 ):
    
    import keras
    import keras.layers as layers
    import tensorflow as tf
        
    
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

def train_selector(load_json=0): # train clasifier model
    
    import model_peformance
    os.system('cls')
    
    tags=[]
    vectors=[]
    
    if not load_json:
        vectors , tags =  model_peformance.main(path='./dataset_black_white/train' , save=0 , train_selector=1 , len_dataset=150)
        with open('./vectors.json' , mode='w') as f:
            json.dump(vectors, f, indent=4)
        
        with open('./tags.json' , mode='w') as f:
            json.dump(tags, f, indent=4)
    else:
        with open('./vectors.json' , mode='r') as f:
            print('loading vectors')
            vectors = json.load(fp=f)
            vectors = np.array(vectors)
        with open('./tags.json' , mode='r') as f:
            print('loading tags')
            tags = json.load(fp=f)
            tags = np.array(tags)
    
    s = ''
    try: # kmeans
        kmeans = classifiers_algorithm.kmeans_signal_classifier(vectors)
        s += f'kmeans SilhouetteScore: { kmeans.SilhouetteScore } \n'
        joblib.dump(kmeans.Model, './model_classifiers/kmeans_model.joblib')
    except Exception as e:
        print('error at kmeans model: ', e)
    
    try: # dbscan
        dbscan = classifiers_algorithm.dbscan_signal_classifier(vectors)
        s += f'"Silhouette Score: { dbscan.SilhouetteScore } , dbscan Adjusted Mutual Info Score: { dbscan.AdjustedMutualInfoScore} , dbscan Completness: {dbscan.Completness} , dbscan Homogeneity: { dbscan.Homogeneity} , dbscan VMeasure: { dbscan.V_Measure }\n'
        joblib.dump(dbscan.Model, './model_classifiers/dbscan.joblib')
    except Exception as e:
        print('error at dbscan model: ', e)
    
    try: # svm
        svm = classifiers_algorithm.svm_signal_classifier(vectors, tags=tags)
        joblib.dump(svm.Model, './model_classifiers/svm.joblib')
        s += f'svm accuracy score:  {svm.AccuracyScore}\n'
    except Exception as e:
        print('error at svm model: ', e)
    
    try: # knn
        knn = classifiers_algorithm.knn_signal_classifier(vectors, tags=tags)
        joblib.dump(knn.Model , './model_classifiers/knn_model.joblib')
        s += f'knn accuracy score:  { knn.AccuracyScore }\n'
    except Exception as e:
        print('error at knn model: ', e)
    
    try: # logitic
        logistic = classifiers_algorithm.logistic_regression_signal_classifier(vectors, tags=tags)
        joblib.dump(logistic.Model, './model_classifiers/logistic.joblib')
        s +=  f'logisitic acurracy score: { logistic.AccuracyScore}\n'
    except Exception as e:
        print('error at logisitic model: ', e)

    with open('./classifier_metric_peformance.txt' , mode='w') as f:
       f.write(s)     
    

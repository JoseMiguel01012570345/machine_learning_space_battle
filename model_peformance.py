import numpy as np
import os
import cv2
import copy
import keras.models
import joblib
import classifiers_algorithm

os.system('cls')

def load_models():
    
    loaded_knn = None
    loaded_kmeans = None
    loaded_dbscan = None
    loaded_svm = None
    loaded_logistic = None
    
    try:
        loaded_knn = joblib.load('./model_classifiers/knn_model.joblib')    
        loaded_kmeans = joblib.load('./model_classifiers/kmeans_model.joblib')
        loaded_dbscan = joblib.load('./model_classifiers/dbscan.joblib')
        loaded_svm = joblib.load('./model_classifiers/svm.joblib')
        loaded_logistic = joblib.load('./model_classifiers/logistic.joblib')
    except:
        pass
    
    return loaded_knn , loaded_kmeans , loaded_dbscan , loaded_svm , loaded_logistic

def selector_peformance( patch:np.array , model_name='' ):
    
    loaded_knn , loaded_kmeans , loaded_dbscan , loaded_svm , loaded_logistic = load_models()
    
    models = {
        'knn': loaded_knn ,
        'kmeans': loaded_kmeans ,
        'dbscan': loaded_dbscan ,
        'svm': loaded_svm ,
        'logisitic': loaded_logistic ,
        
    }
    
    m = []
    for v in patch:
        r = classifiers_algorithm.get_signal_features( v )

        q= np.array([r.Average , r.Bandwidth , r.Centroid , r.Contrast])

        q = np.where(~np.isnan(q), q, 0)
        m.append(q)

    m = np.array(m)
    
    model = models[model_name]
    if model is not None and model_name != 'dbscan':
        return model.predict(m)
    elif model_name == 'dbscan':
        return model.fit_predict(patch)
    
    print('model is not avaliable')
    
    return None

def use_model( x:np.array= None  , model_list:list = [] , model=-1 ):
        
    num_patches = 100
    best_similarity = 1e305
    best_img = np.array([])
    
    best_version = 0
    
    if model != -1:
        output = selector_peformance(patch=x , model_name=model)
        return output
    
    for v in range(num_patches): # brute force
        
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
            best_version = v
        
    return best_img , best_version

def main(path='./dataset_black_white/validation' , save= 1 , train_selector=0 , len_dataset=-1):

    num_models = 100

    validation_img = []
    for i in os.listdir(path=path):
        validation_img.append( np.where(cv2.imread( path + '/' + i , cv2.IMREAD_GRAYSCALE ) > 128 , 1.0 , 0.0 ))

    if len_dataset != -1:
        validation_img = validation_img[:len_dataset]
    
    count = 0
    _input_ = None
    model_list = [ keras.models.load_model( f'./model_pattern_recog/model_v{i}' )  for i in range( num_models )  ]
        
    offset = 100
    dataset = []
    tags = []
    # model_name = 'kmeans'
    model_name = -1
    for img in validation_img:
        print('generating...') 
        _input_ = copy.deepcopy( img )
        for patch_row in range(int(img.shape[1] / offset)):
            for patch_column in range(int(img.shape[0] / offset)):
                    
                    sample = np.ones((100,100))
                    for i in range( 100 ): # get patch
                        
                        k = patch_row * offset + i
                        if k >= img.shape[0]: break
                        
                        for j in range( 100 ):
                            l= patch_column* offset + j
                        
                            if l >= img.shape[0]: break
                            sample[ j , i  ] = img[l,k]
                    
                    if train_selector:
                        vector = np.zeros((num_models ,))
                        sample , version = use_model( x=sample , model_list=model_list )
                        vector[version] = 1
                        dataset.append( list( sample.flatten()) )
                        tags.append(version)
                    else:
                        model_version = use_model( x=sample , model_list=model_list , model=model_name )
                        model_version = np.argmax(model_version)
                        sample = np.reshape(sample , (100,100))
                        sample = np.expand_dims(sample , 0)
                        sample = np.expand_dims(sample , -1)
                        y = model_list[model_version]( sample )
                        
                        y = np.squeeze(y) # adjust output model for representation
                        y = y / np.max(y)
                        y = np.clip( y , 0.0 , 1.0 )
                        y = np.where( y >= (np.max(y) + np.min(y))/2 , 1.0 , 0.0 )
                        y = np.reshape( y , (100,100) )
                        sample = y
                        
                    if not train_selector:
                        for i in range( 100 ): # set patch
                            
                            k = patch_row * offset + i
                            if k >= img.shape[0]: break
                            
                            for j in range( 100 ):
                                l= patch_column * offset + j
                            
                                if l >= img.shape[0]: break
                                img[l,k] = sample[ j , i  ]
                    
        count += 1
        print(count)
        if save:
            cv2.imwrite( f'./gallery/gallery_{model_name}/{count}I.jpg' , img=np.where( _input_ * 255.0 > 128.0 , 255.0 , 0.0) ) # write input in gallery folder
            cv2.imwrite( f'./gallery/gallery_{model_name}/{count}O.jpg' , img=np.where(img* 255.0 > 128.0 , 255.0 , 0.0) ) # write ouput in gallery folder
        
    return dataset , tags
    
main(len_dataset=10)
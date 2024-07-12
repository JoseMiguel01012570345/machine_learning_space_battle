"""
signal

a module for work with signals
"""
from signals.signal_processing import matrix2canonics_coefs,SpectralFeatures,spectral_centroid,spectral_bandwidth,get_signal_features,spectral_rolloff,spectral_contrast,spectral_flatness
import numpy as np
from pathlib import Path
import json

def get_samples_features_vectors(samples):
    for sample in samples:
        v = get_signal_features(sample)
        yield np.array([v.Centroid,v.Bandwidth,v.Flatness,v.Average])
        pass
    pass

def SaveSamplesFeatures(samples,good=True,path='./maps_features'):
    root = Path(path)
    if not root.exists or not root.is_dir():
        raise Exception('la ruta no es valida')
    cont = 0
    for f in root.iterdir():
        cont += 1
        pass
    for sample in samples:
        features = get_signal_features(sample)
        file_name = f'map{cont}.json'
        file = open(f'{root.joinpath(file_name)}','w')
        data = {
            'good':good,
            'features':[features.Centroid,features.Bandwidth,features.Flatness,features.Average]
        }
        file.write(json.dumps(data))
        file.close()
        cont += 1
        pass
    pass

def LoadSamplesFeatures(path='./maps_features'):
    root = Path(path)
    if not root.exists or not root.is_dir():
        raise Exception('la ruta no es valida')
    for f in root.iterdir():
        if not f.suffix == '.json': continue
        file = open(f'{f}','r')
        data = json.loads(file.read())
        file.close()
        yield data['good'],data['features'],f.name
        pass
    pass
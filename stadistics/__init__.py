"""
stadistics

a module to make stadistics analysis about a dataset saved in json format
"""
import json
import numpy as np
from scipy.linalg import eig
from os import getcwd
from pathlib import Path
from stadistics.data import load_file,load,DATA_PATH
from stadistics.visualization2D import DispersionGraphic,Histogram,LinearGraphic,BlocksLinearGraphic,BarGraphic,BlocksDispersionGraphic,BlocksSpaceRelationGraphic,BoxPlotGraphic
from stadistics.visualization3D import DispersionGraphic3D,BlocksDispersionGraphic3D,LinearGraphic3D,BlocksLinearGraphic3D
from stadistics.preprocessing import BlocksToSignal,MatrixFFT,VisualizeFrequency
from stadistics.data import change_path as ChangePathToData
from stadistics.visualization2D import change_path as ChangePathToImages2D
from stadistics.visualization3D import change_path as ChangePathToImages3D
from stadistics.visualization2D import IMAGE_PATH as IMAGE2D_PATH
from stadistics.visualization3D import IMAGE_PATH as IMAGE3D_PATH
from scipy.fft import fft

# DATASET = load()

def MatrixEigenvaluesSignal(matrix):
    """
    returns a signal made form the eigenvalues of the given matrix
    """
    eigenvalues,_ = eig(matrix)
    return eigenvalues
    

def BuildMatrix(blocks,shape):
    """
    build the maps given the blocks-coordinates
    shape -> tuple(int,int)
    blocks -> list(int,int)
    """
    matrix = np.zeros(shape)
    for block in blocks:
        matrix[block[0],block[1]] = 1
        pass
    return matrix

def SaveMatrixSpectresImage(matrix_array,shape):
    """
    saves the spectres-images for the set of matrix
    shape -> tuple with the sizes of the matrix
    """
    fig = 1
    for m in matrix_array:
        matrix = BuildMatrix(m,shape)
        MatrixFFT(matrix,1,1000,show=False,save=True,title=f'fig{fig}',normalize=True)
        fig += 1
        pass
    pass

def SaveMatrixSpectres(matrix_array,shape,**kwargs):
    """
    saves the spectres of the matrix sets given in a json file
    keys suppourted:
        path: str -> most be the path to the folder where the data will be stored
        fname: str -> the name of the file where the data will be stored
        shape: tuple -> the dimensions of the matrixs to store
    """
    path=Path(getcwd())
    fname='spectres'
    
    if 'path' in kwargs.keys():
        path = Path(kwargs['path'])
        pass
    if 'fname' in kwargs.keys():
        fname = kwargs['fname']
        pass
    if not path.exists:
        raise Exception('la ruta no existe')
    file = path.joinpath(fname)
    fig = 1
    matrixs = {'shape': shape}
    for m in matrix_array:
        matrix = BuildMatrix(m,shape)
        fft = MatrixFFT(matrix,1,1000,show=False,title=f'fig{fig}',normalize=True)
        values = [str(value) for value in fft]
        matrixs[f'fig{fig}'] = values
        fig += 1
        pass
    json_data = json.dumps(matrixs)
    writer = open(f'{file}.json','w')
    writer.write(json_data)
    writer.close()
    pass

def SaveSignalsSpectres(signals,path):
    root = Path(path)
    if not root.exists or root.is_dir():
        raise Exception('la ruta especificad no es valida')
    for i,signal in enumerate(signals):
        freq = fft(signal)
        data = []
        for v in freq:
            data.append((v.real,v.imag))
            pass
        file_path = root.joinpath(f'map{i}.json')
        file = open(file_path,'w')
        file.write(json.dumps(data))
        file.close()
        pass
    pass

def LoadSignalsSpectres(path):
    root = Path(path)
    if not root.exists or root.is_file():
        raise Exception('la ruta especificada no es valida')
    freqs = []
    for f in root.iterdir():
        file = open(f'{f.resolve()}','r')
        data = json.loads(file.read())
        file.close()
        freq = []
        for value in data:
            real = str(value[0])
            imag = str(value[1])
            n = ''
            if imag.startswith('-'):
                n = f'{real}{imag}j'
                pass
            else:
                n = f'{real}+{imag}j'
                pass
            freq.append(np.complex128(n))
            pass
        freqs.append(freq)
        pass
    return np.array(freqs)
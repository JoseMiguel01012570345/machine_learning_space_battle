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

DATASET = load()

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

def LoadMatrixSpectres(fname):
    """
    loads the spectres of the matrixs storeds in the given path
    """
    path = Path(fname)
    if not path.exists or not path.is_file():
        raise Exception('la ruta especificada no es valida')
    if not path.suffix == '.json':
        raise Exception('formato no valido')
    reader = open(f'{path}','r')
    content = reader.read()
    reader.close()
    try:
        json_data = json.loads(content)
        shape = tuple(json_data['shape'])
        maps = {}
        for key in json_data.keys():
            if not key == 'shape':
                maps[key] = [np.complex128(value) for value in json_data[key]]
                pass
            pass
        return shape,maps
    except Exception as ex:
        print('Error, formato no sorportado: {ex}')
        return (-1,-1),{}
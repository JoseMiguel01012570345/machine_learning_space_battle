"""
visualization3D

a module to visualizace data in 3 dimensions
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from os import getcwd

ROOT_PATH = getcwd()
IMAGE_PATH = Path(ROOT_PATH).joinpath('images')

def change_path(path):
    path = Path(path)
    if not path.exists:
        raise Exception('la ruta no existe')
    IMAGE_PATH = path
    pass

def DispersionGraphic3D(x,y,z,show=True,save=False,title='fig'):
    """
    generate a 3D dispersion graphic
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x,y,z,c=z,cmap='viridis',marker='o')
    ax.set_xlabel('Eje x')
    ax.set_ylabel('Eje y')
    ax.set_zlabel('Eje z')
    plt.title(title)
    if save:
        path = IMAGE_PATH.joinpath(title).resolve()
        plt.savefig(path)
        pass
    if show:
        plt.show()
        pass
    plt.close('all')
    pass

def LinearGraphic3D(x,y,z,show=True,save=False,title='fig'):
    """
    generate a linear graphic in 3 dimensions
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(x,y,z,color='blue')
    ax.set_xlabel('Eje x')
    ax.set_ylabel('Eje y')
    ax.set_zlabel('Eje z')
    plt.title(title)
    if save:
        path = IMAGE_PATH.joinpath(title).resolve()
        plt.savefig(path)
        pass
    if show:
        plt.show()
        pass
    plt.close('all')
    pass

def BlocksDispersionGraphic3D(blocks,show=True,save=False,title='fig'):
    """
    generate a 3D dispersion graphic for the blocks
    """
    
    freq = {}
    X = [block[0] for block in blocks]
    Y = [block[1] for block in blocks]
    for block in blocks:
        if not (block[0],block[1]) in freq.keys():
            freq[(block[0],block[1])] = 1
            pass
        else:
            freq[(block[0],block[1])] += 1
            pass
        pass
    Z = [freq[(X[i],Y[i])] for i in range(len(X))]
    DispersionGraphic3D(X,Y,Z,show,save,title)
    pass

def BlocksLinearGraphic3D(blocks,show=True,save=False,title='fig'):
    """
    generate a 3D linear graphic for the blocks
    """
    
    freq = {}
    X = [block[0] for block in blocks]
    Y = [block[1] for block in blocks]
    for block in blocks:
        if not (block[0],block[1]) in freq.keys():
            freq[(block[0],block[1])] = 1
            pass
        else:
            freq[(block[0],block[1])] += 1
            pass
        pass
    Z = [freq[(X[i],Y[i])] for i in range(len(X))]
    LinearGraphic3D(X,Y,Z,show,save,title)
    pass
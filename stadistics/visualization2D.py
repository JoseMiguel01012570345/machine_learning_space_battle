"""
visualization2D

a module to visualizace data in 2 dimensions
"""
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

def DispersionGraphic(x,y,show=True,save=False,title='fig',xlabel='x',ylabel='y'):
    """
    generate a dispersion-graphic with the x, and y variables
    x,y -> arrays
    """
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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

def Histogram(data,show=True,save=False,title='fig'):
    """
    generate an histogram visualization for the given data
    """
    plt.hist(data,bins=30,rwidth=0.7,align='mid',label='blocks')
    plt.title(title)
    plt.legend(loc='upper right')
    if save:
        path = IMAGE_PATH.joinpath(title).resolve()
        plt.savefig(path)
        pass
    if show:
        plt.show()
        pass
    plt.close('all')
    pass

def LinearGraphic(x,y,show=True,save=False,title='fig'):
    """
    generate a linear graphic with the given data
    """
    plt.plot(x,y)
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

def BarGraphic(types,values,show=True,save=False,title='fig'):
    """
    generate a bar-graphic
    """
    plt.bar(types,values)
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

def BoxPlotGraphic(data,show=True,save=False,title='fig'):
    plt.boxplot(data)
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

def BlocksDispersionGraphic(blocks,show=True,save=False,title='fig'):
    """
    generate a dispersion grafic for the blocks coordinates
    """
    X = [block[0] for block in blocks]
    Y = [block[1] for block in blocks]
    DispersionGraphic(X,Y,show,save,title)
    pass

def BlocksLinearGraphic(blocks,show=True,save=False,title='fig'):
    """
    generate a linear graphic for the given list of blocks
    """
    freq = {}
    blocks_ids = {}
    blocks_pos = 0
    for block in blocks:
        if not tuple(block) in freq.keys():
            freq[tuple(block)] = 1
            blocks_ids[blocks_pos] = tuple(block)
            blocks_pos += 1
            pass
        else:
            freq[tuple(block)] += 1
            pass
        pass
    
    X = list(blocks_ids.keys())
    Y = [freq[tuple(blocks_ids[block])] for block in X]
    LinearGraphic(X,Y,show,save,title)
    pass

def BlocksSpaceRelationGraphic(blocks,total,show=True,save=False,title='fig'):
    """
    generate a graphic for the blocks-nonblock relation
    """
    cat = ['block','space']
    values = [len(blocks),total - len(blocks)]
    BarGraphic(cat,values,show,save,title)
    pass
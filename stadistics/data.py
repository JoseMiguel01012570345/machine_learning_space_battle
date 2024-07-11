"""
data

a module to read and write data in json format for the later analysis
"""
import json
import numpy as np
from pathlib import Path
from os import getcwd

ROOT_PATH = getcwd()
DATA_PATH = Path(ROOT_PATH).joinpath('MapsSet')

def change_path(path):
    """
    changes the path to the data
    """
    DATA_PATH = Path(path)
    pass

def parse_data(data):
    new_data = data.replace('(','').replace(')','').split(',')
    return int(new_data[0]),int(new_data[1])

def load():
    """
    load all the data and returns it in a dictionary
    """
    maps = {}
    for file in DATA_PATH.iterdir():
        if file.is_file() and file.suffix == '.json':
            name = file.name
            pos = name.index(file.suffix)
            key_name = name[:pos]
            data = load_file(file.resolve())
            maps[key_name] = np.array([parse_data(value) for value in data])
        pass
    return maps

def load_file(path):
    """
    load the data into the given file
    WARNING! the file most be a json file 
    """
    reader = open(path,'r')
    json_data = reader.read()
    reader.close()
    return json.loads(json_data)

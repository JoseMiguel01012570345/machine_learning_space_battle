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
            print( f"\033[1;32m {key_name} \033[0m ")
            maps[key_name] =tokenize_data( data=data , key_name=key_name )
        pass
    return maps

def tokenize_data( data:dict={} ,key_name:str = ""):

    len_row = data["row"]
    len_column = data["column"]
    maps = {}
    
    simple_map = []
    
    for i in range(len_row):
        
        column = data[ str(i) ].split(" ")[1:]
        
        for j in column :
            
            pair = ( i , int(j)  )
            simple_map.append( pair )
        
    return np.array(maps)

def load_file(path):
    """
    load the data into the given file
    WARNING! the file most be a json file 
    """
    reader = open(path,'r')
    json_data = reader.read()
    reader.close()
    return json.loads(json_data)

load()

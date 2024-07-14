"""
data

a module to read and write data in json format for the later analysis
"""
import json
import numpy as np
from pathlib import Path
from os import getcwd
import os

ROOT_PATH = "../"
DATA_PATH = Path(ROOT_PATH).joinpath('MapsSet')

def change_path(path):
    """
    changes the path to the data
    """
    DATA_PATH = Path(path)
    pass

def load(total=None):
    """
    load all the data and returns it in a dictionary
    """
    maps = {}
    for file in DATA_PATH.iterdir():
        if file.is_file() and file.suffix == '.json':
            if total:
                total -= 1
                pass
            name = file.name
            pos = name.index(file.suffix)
            key_name = name[:pos]
            data = load_file(file.resolve())
            print( f"\r\033[1;32m {key_name} \033[0m ")
            # os.system("cls")   
            maps[key_name] =tokenize_data( data=data , key_name=key_name )
            if total == 0: break 
            
        pass
    return maps

def tokenize_data( data:dict={} ,key_name:str = ""):

    len_row = data["row"]
    
    simple_map = []
    
    for i in range(len_row):
        
        column = data[ str(i) ].split(" ")[1:]
        
        for j in column :
            
            pair = ( i , int(j)  )
            simple_map.append( pair )
        
    return np.array(simple_map)

def load_file(path):
    """
    load the data into the given file
    WARNING! the file most be a json file 
    """
    reader = open(path,'r')
    json_data = reader.read()
    reader.close()
    return json.loads(json_data)

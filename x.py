import os
while True:
    try:
        c= os.system('git fetch')
        if c ==0: break
    except:
        pass
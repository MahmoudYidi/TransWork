import csv
import numpy as np 

def get_rot_tran(id, data):
    print(id)
    output = data[id-1][:]
    output = [float(x) for x in output]
    output = np.array([output])
    return output   



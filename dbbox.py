import numpy as np
def bboxcorrect(bbox,fac1,fac2):

    factor = fac1 
    #[A,B] = findCenter(bbox)
    bboxi = np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
    ])
    ### X component ####
    bboxi[0][0] = bbox[0][0] + factor
    bboxi[1][0] = bbox[1][0] + factor
    bboxi[2][0] = bbox[2][0] + factor
    bboxi[3][0] = bbox[3][0] + factor
    factor = fac2
    ##### Y component ###
    bboxi[0][1] = bbox[0][1] + factor
    bboxi[1][1] = bbox[1][1] + factor
    bboxi[2][1] = bbox[2][1] + factor
    bboxi[3][1] = bbox[3][1] + factor

    
    return bboxi
    

def findCenter(bbox):
    A = ((bbox[1][0] - bbox[0][0]) / 2) + bbox[0][0]
    B = ((bbox[2][1] - bbox[1][1]) / 2) + bbox[1][1]

    return [A,B]




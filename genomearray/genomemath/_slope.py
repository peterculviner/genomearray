import numpy as np

def _vectorslope(y_vectors):
    """ Accepts an array of arrays and calculates """
    x = np.reshape(np.arange(np.shape(y_vectors)[0]).astype(float),(-1,1))
    y = np.asarray(y_vectors).astype(float)
    slope_vector = ((len(x)*np.sum(x*y,axis=0)-np.sum(x,axis=0)*np.sum(y,axis=0))/
                    (len(x)*np.sum(np.square(x),axis=0)-np.square(np.sum(x,axis=0))))
    return slope_vector

def rollingSlope(input_data, n_positions):
    # roll arrays to get slope of n_positions in a forward direction
    rolled_data = []
    for left,right in zip(range(n_positions)[::-1],range(n_positions)):
        rolled_data.append(np.r_[np.zeros(left)+np.nan,input_data,np.zeros(right)+np.nan])
    slopes = _vectorslope(np.asarray(rolled_data))[n_positions-1:]
    return slopes
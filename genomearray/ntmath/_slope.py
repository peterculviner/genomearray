import numpy as np

def _vectorslope(y_vectors):
    """ Accepts an array of arrays and calculates least squares slope. """
    x = np.reshape(np.arange(np.shape(y_vectors)[0]).astype(float),(-1,1))
    y = np.asarray(y_vectors).astype(float)
    slope_vector = ((len(x)*np.sum(x*y,axis=0)-np.sum(x,axis=0)*np.sum(y,axis=0))/
                    (len(x)*np.sum(np.square(x),axis=0)-np.square(np.sum(x,axis=0))))
    return slope_vector

def _vectorrollingslope(input_array, n_positions):
    # roll arrays to get slope of n_positions in a forward direction
    rolled_array = []
    for left,right in zip(range(n_positions)[::-1],range(n_positions)):
        rolled_array.append(np.r_[np.zeros(left)+np.nan,input_array,np.zeros(right)+np.nan])
    out = _vectorslope(np.asarray(rolled_array))[n_positions-1:]
    return out

def rollingslope(input_array, slope_distance, slope_position):
    """ Returns the rolling least squares slope across the genome.

        The input_array is presumed to be shape (2, genome_length) and least squares slope is
        calculated across a distance of slope_distance. slope_position should be set to 5 or
        3, meaning slopes will stored to either the 5' or 3' end of the calculated region.
        
        Parameters:
        ----------
        input_array : numpy array of shape (2, len(genome))
            Slope will be calculated across this array.

        slope_distance : int
            Number of nucleotides across which slope will be calculated.
        
        slope_position : '5_prime' or '3_prime'
            Position (5' or 3') to which to record slope.

        Returns:
        ----------
        out : numpy array of same shape as input_array
            Slopes are stored at 5' or 3' ends of slope_distance. Positions for which slope
            could not be calculated are assigned np.nan as a placeholder to maintain input shape.
        """
    # caculate slope for the positive strand
    positive_slopes = _vectorrollingslope(input_array[0], slope_distance)
    # calculate for the negative strand, reverse before calculation to preserve 5' -> 3' directionality
    negative_slopes = _vectorrollingslope(np.flip(input_array[1],0), slope_distance)
    # if using a 3' slope, roll the array to account for this
    if slope_position == '3_prime':
        # positive_slopes = np.r_[np.zeros(slope_distance-1)+np.nan,positive_slopes[:-slope_distance+1]]
        positive_slopes = np.roll(positive_slopes,slope_distance-1)
        # negative_slopes = np.r_[np.zeros(slope_distance-1)+np.nan,negative_slopes[:-slope_distance+1]]
        negative_slopes = np.roll(negative_slopes,slope_distance-1)
    elif slope_position != '5_prime': # check if user inputted actual slope position
        raise ValueError("Choose a valid slope position (either '5_prime' or '3_prime').")
    # return slopes in a genome-shaped array
    out = np.asarray([positive_slopes, np.flip(negative_slopes,0)])
    return out
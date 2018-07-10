import numpy as np
import genomearray as ga

def flatregions(input_array, slope_distance, slope_position, 
                array_mask = None, lower_percentile = None, upper_percentile = None):
    """ Finds regions on an input_array with a relatively flat slope.

        Takes the 5' -> 3' slope of the input_array over a distance of slope_distance and returns a
        region mask of positions between lower_percentile and upper_percentile of slope. Any positions
        set to False in the  array_mask will not be used in calculation of percentiles and will not 
        be returned.
        
        Parameters:
        ----------
        input_array : numpy array of shape (2, len(genome))
            Array on which slopes and percentiles are calculated.

        slope_distance : int
            Nucleotide distance over which to calculate slope. Argument to ga.ntmath.rollingslope.

        slope_position : '5_prime' or '3_prime'
            Position (5' or 3') to which to record slope. Argument to ga.ntmath.rollingslope.

        array_mask : boolean numpy array of same shape as input_array
            Positions which are False in this array will be used to calculate slope, but will not
            be used to calculate percentile and will not be returned in output.

        lower_percentile : float or integer (optional)
            Minimum percentile of slopes to consider for membership in the returned flat regions.
            If None, no lower limit.

        upper_percentile : float or integer (optional)
            Maximum percentile of slopes to consider for membership in the returned flat regions.
            If None, no upper limit.

        Returns:
        ----------
        output_array : boolean numpy array of same shape as input_array
            An array of relatively 'flat' regions determined by taking slopes across input_array.
    """
    # calculate slopes across the input_array
    slope_array = ga.ntmath.rollingslope(input_array, slope_distance, slope_position)
    # mask on the slope array all positions to not be considered to np.nan
    slope_array[~np.asarray(array_mask).astype(bool)] = np.nan
    # find lower and upper percentiles
    if lower_percentile is not None:
        lower_threshold = np.nanpercentile(slope_array, lower_percentile)
    else:
        lower_threshold = np.nanmin(slope_array)
    if upper_percentile is not None:
        upper_threshold = np.nanpercentile(slope_array, upper_percentile)
    else:
        upper_threshold = np.nanmax(slope_array)
    # return all positions between (inclusive) upper and lower percentiles
    return np.all([slope_array >= lower_threshold, slope_array <= upper_threshold], axis=0)

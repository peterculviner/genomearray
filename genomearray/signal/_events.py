import numpy as np
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy.ndimage.filters import gaussian_filter1d
import genomearray as ga

def extrema(input_array, extrema_type = 'min', output_mask = None, smooth_sigma = None, search_nt = None):
    """ Detects and returns extrema positions in genome-shaped arrays.

        Searches input_array (assumed to be genome-shaped) for local maxima or minima (see
        extrema_type) and returns an array of positions as [[strand, nt_pos]....] as well
        as the value at that position.
        
        Parameters:
        ----------
        input_array : numpy array of shape (2, len(genome))
            Local extrema are found and recorded from this array.

        extrema_type : 'min' (default) or 'max'
            Type of extrema to search for.
        
        output_mask : None (default) or boolean array of same shape as input_array
            If None, function extrema at all positions are considered and returned. Otherwise, user
            defined mask is used and only positions set to True are returned.

        smooth_sigma : None (default) or sigma scipy.ndimage.filters.gaussian_filter1d (float) 
            If None, no smoothing is conducted. If a float, this value is passed to the gaussian
            smoothing function as the sigma value for smoothing.

        search_nt : None (default) or int
            Distance to search (in both directions) for true extrema. Useful if smoothing 
            arguements have been passed as search is conducted on original (not smoothed)
            array.

        Returns:
        ----------
        positions : positions of extrema, numpy array of shape (n extrema, 2)
            Found extrema positions are returned with first column denoting strand and second column
            denoting genomic position.
        
        values : values of extrema on input_array, numpy array of shape (n extrema,)
            Potentially useful for downstream sorting of events by value.
    """
    if smooth_sigma is None: # if without smoothing, use input array as search array 
        search_array = input_array.copy()
    else: # otherwise pass smooth_sigma to gaussian_filter1d as the sigma
        search_array = gaussian_filter1d(input_array, smooth_sigma)
    # search for extrema
    if extrema_type == 'min':
        extrema_pos = np.asarray(argrelmin(search_array, axis=1))
    elif extrema_type == 'max':
        extrema_pos = np.asarray(argrelmax(search_array, axis=1))
    else:
        raise ValueError('Unhandled extrema type.')
    # further refine placement of extrema on original, unsmoothed array using regionfunc if search_nt is not None
    if search_nt is not None:
        if extrema_type == 'min':
            # check surrounding regions (+/- search_nt) for any lower/higher extrema positions
            relative_extrema = ga.regionfunc(np.argmin, extrema_pos.T, input_array, addl_nt = (search_nt, search_nt), wrt = 'genome')
        elif extrema_type == 'max':
            relative_extrema = ga.regionfunc(np.argmax, extrema_pos.T, input_array, addl_nt = (search_nt, search_nt), wrt = 'genome')
        else:
            raise ValueError('Unhandled extrema type.')
        extrema_pos = np.asarray([extrema_pos.T[:,0],
                                  extrema_pos.T[:,1] + relative_extrema - search_nt]) # convert relative extrema to actual genomic position
    # get values for position on original input_array, prepare for final output
    positions = extrema_pos.T
    values    = input_array[tuple(extrema_pos)]
    # remove any extrema which are in False positions on output_mask
    if output_mask is not None:
        positions = positions[output_mask[tuple(extrema_pos)]]
        values = values[output_mask[tuple(extrema_pos)]]
    return positions, values

def eventdpos(primary_pos, secondary_pos, maximum_distance, direction='5_prime', collpase_regions=True):
    """ Finds regions based on primary_pos and secondary_pos arrays.

        Starting with an array of primary_pos, attempts to define regions with secondary_pos within
        maximum_distance in the direction indicated. Overlapping regions are collapsed into single
        regions by default.
        
        Parameters:
        ----------
        primary_pos : array of positions, numpy array of shape (n positions, 2)
            List of possible positions to seed event regions.

        secondary_pos : array of positions, numpy array of shape (n positions, 2)
            List of positions to search on starting from primary positions.

        maximum_distance : distance in nt to search, int
            Maximum distance to search from primary_pos to find a region. The shortest possible
            region is returned.

        direction : direction to search from primary_pos, '5_prime' (default) or '3_prime'
           Search direction from primary_pos.

        collapse_regions : collapses overlapping regions, True (default) or False
            If True, collapses all overlapping regions into a single region extending to the
            leftmost coordinate and rightmost coordinate of all overlapping regions. Otherwise
            returns region list ignoring overlaps.

        Returns:
        ----------
        events : array of regions, numpy array of shape (n regions, 3)
            First column denotes strand, second column denotes left position (inclusive), third
            column denotes right position (inclusive).

        """
    pass

def eventdyperx(input_array, position_array, dy, maximum_distance, collapse_regions=True):
    """ Finds regions based on shape of a peak or valley surrounding user-provided positions.

        For all positions in position_array, attempts to define a region by searching upstream (5')
        and downstream (3') of the defined position by maximum_distance nt. A region is recorded if
        the dy (measured from the starting position to the current position) is met within
        maximum_distance on both sides.
        
        Parameters:
        ----------
        input_array : numpy array of shape (2, len(genome))
            Peaks or valleys are defined across the value (y) in this array.

        position_array : array of positions, numpy array of shape (n positions, 2)
            List of possible positions to seed event regions.

        dy : minimum change in dy required to define a region, tuple: (float, float)
            First member of tuple defines distance upstream (5') to look, second downstream. First
            position which meets the requisite dy is recorded in either direction.

        maximum_distance : maximum distance to search for requsite dy, tuple: (int, int)
            As with dy, first member is upstream, second member is downstream. Positions beyond
            maximum_distnace from position_array are not considered.

        collapse_regions : collapses overlapping regions, True (default) or False
            If True, collapses all overlapping regions into a single region extending to the
            leftmost coordinate and rightmost coordinate of all overlapping regions. Otherwise
            returns region list ignoring overlaps.

        Returns:
        ----------
        events : array of regions, numpy array of shape (n regions, 3)
            First column denotes strand, second column denotes left position (inclusive), third
            column denotes right position (inclusive).

        """
    events = [] # list for storing events meeting criteria
    for pos in position_array:
        strand, position = pos # unpack strand and position
        value = input_array[strand,position] # get value at position
        try:
            if strand == 0:
                # try to find first position meeting delta criteria on the genomic left
                left_slice = input_array[strand,position-maximum_distance[0]:position+1]
                if dy[0] > 0: # positive change
                    left = np.where(left_slice-value >= dy[0])[0][-1]
                else: # negative change
                    left = np.where(left_slice-value <= dy[0])[0][-1]
                # try to find a position meeting delta criteria on the genomic right
                right_slice = input_array[strand,position:position+maximum_distance[1]+1]
                if dy[1] > 0: # positive change
                    right = np.where(right_slice-value >= dy[1])[0][0]
                else: # negative change
                    right = np.where(right_slice-value <= dy[1])[0][0]
                # convert relative left / right to genomic left / right
                genomic_left = position-maximum_distance[0]+left
                genomic_right = position+right
            elif strand == 1:
                # try to find first position meeting delta criteria on the genomic left
                left_slice = input_array[strand,position-maximum_distance[1]:position+1]
                if dy[1] > 0: # positive change
                    left = np.where(left_slice-value >= dy[1])[0][-1]
                else: # negative change
                    left = np.where(left_slice-value <= dy[1])[0][-1]
                # try to find a position meeting delta criteria on the genomic right
                right_slice = input_array[strand,position:position+maximum_distance[0]+1]
                if dy[0] > 0: # positive change
                    right = np.where(right_slice-value >= dy[0])[0][0]
                else: # negative change
                    right = np.where(right_slice-value <= dy[0])[0][0]
                # convert relative left / right to genomic left / right
                genomic_left = position-maximum_distance[1]+left
                genomic_right = position+right
            else:
                raise ValueError('strand must be 0 or 1.')
            # add region to events list
            events.append([strand, genomic_left, genomic_right])
        except IndexError:
            continue # an index error indicates a failure to meet criteria on one or both sides
    if collapse_regions:
        return ga.concatregions(np.asarray(events))
    else:
        return np.asarray(events)
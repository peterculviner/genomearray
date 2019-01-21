import numpy as np

def genomeslice(input_array, strand, left, right, wrt = '5_to_3'):
    """Return 5' -> 3' slice of genome array based on inclusive coordinantes."""
    if left > right: # empty slice case
        return np.asarray([])
    elif (strand == 0) or (wrt is 'genome'):
        return input_array[strand,left:right+1]
    elif (strand == 1) and (wrt is '5_to_3'):
        return np.flip(input_array[strand,left:right+1],axis=0)
    else:
        raise ValueError("Unhandled strand {0 or 1} or wrt {'genome' or '5_to_3'} value.")

def regionslice(regions, input_array, addl_nt = (0,0), wrt='5_to_3'):
    """ Returns the slice of given regions on input_array, + / - addl_nt.

        For each region in regions array, get the slice across the inclusive coordiniates on the 
        input_array with any additional bases on the left and right sides defined by addl_nt.
        The wrt kwarg controls if the slice is returned in the left -> right orientation or in the
        5' -> 3' orientation (right -> left for the second strand).
        
        Parameters:
        ----------
        regions : array-like, shape (n regions, 2) or (n regions, 3)
            If the array has 2 columns, the first column is presumed to be strand (0 or 1) and the
            second column is used to define a position (i.e. slice would return a single nt if no
            addl_nt are provided on either side). If the array has 3 columns, the third defines the
            right position (inclusive) of the slice.

        input_array : numpy array, shape (2, len genome)
            The genome-shaped data from which to slice.

        addl_nt : (0,0) (default) or any tuple shape (2,) of int
            The first term is the number of nt to add to the left (in genome or 5' in 5' to 3'
            orientation). The second term is the same for the right (or 3').

        wrt : '5_to_3' (default) or 'genome'
            The orientation in which to feed the slices into the input_function. genome orientation
            is always left -> right, regardless of strand. 5_to_3 reverses the direction if the
            slice is on the second strand to maintain 5' -> 3' orientation.

        Returns:
        ----------
        out : list of same length as regions
            Slices of input_array defined by regions with additional accoutrements defined by
            addl_nt and wrt.

        """
    # handle as special case of genomearray.regionfunc where the function returns the input
    return regionfunc(lambda x: x, regions, input_array, addl_nt = addl_nt, wrt = wrt)

def regionfunc(input_function, regions, input_array, addl_nt = (0,0), wrt = '5_to_3'):
    """ Return the output of a function across the given regions on input_array, + / - addl_nt.

        For each region in regions array, run the input_function across the inclusive coordiniates
        on the input_array with any additional bases on the left and right sides defined by addl_nt.
        The wrt kwarg controls if second strand regions are run through the function in the left ->
        right orientation or in the 5' -> 3' orientation (right -> left for the second strand).
        
        Parameters:
        ----------
        input_function : function
            A user defined function to run on all slices.

        regions : array-like, shape (n regions, 2) or (n regions, 3)
            If the array has 2 columns, the first column is presumed to be strand (0 or 1) and the
            second column is used to define a position (i.e. slice would return a single nt if no
            addl_nt are provided on either side). If the array has 3 columns, the third defines the
            right position (inclusive) of the slice.

        input_array : numpy array, shape (2, len genome)
            The genome-shaped data from which to slice.

        addl_nt : (0,0) (default) or any tuple shape (2,) of int
            The first term is the number of nt to add to the left (in genome or 5' in 5' to 3'
            orientation). The second term is the same for the right (or 3').

        wrt : '5_to_3' (default) or 'genome'
            The orientation in which to feed the slices into the input_function. genome orientation
            is always left -> right, regardless of strand. 5_to_3 reverses the direction if the
            slice is on the second strand to maintain 5' -> 3' orientation.

        Returns:
        ----------
        out : list of same length as regions
            Output of input_function across regions defined by regions with additional accoutrements
            defined by addl_nt and wrt.

        """
    # check if it only has one position term, duplicate it if it does
    if regions.shape[1] == 2:
        regions = np.asarray([regions[:,0],regions[:,1],regions[:,1]]).T
    out = []
    for reg in regions:
        strand, left, right = reg
        if (wrt is '5_to_3') and (strand == 1): # if second strand and 5' -> 3', left and right definitions are swapped
            left = left - addl_nt[1]
            right = right + addl_nt[0]
        else:
            left = left - addl_nt[0]
            right = right + addl_nt[1]
        try:
            # to get arrays, use genomeslice function to ensure directional (i.e. 5' -> 3') input_functions work as intended
            left = np.maximum(0,left)
            right = np.minimum(input_array.shape[1],right)
            out.append(input_function(genomeslice(input_array, strand, left, right, wrt=wrt)))
        except:
            out.append(np.nan) # if function raises an exception, add np.nan to the list
    return out

def _splitregion(region_len, window_len, stride):
    n_steps = (region_len - window_len) / stride + 1
    lefts = np.asarray([i*stride for i in range(n_steps)])
    rights = lefts+window_len-1
    remainder = region_len - rights[-1] - 1
    return lefts + remainder/2, rights + remainder/2

def splitregions(region_array, window_len, stride):
    """ Split regions into sub-regions with a given window length and stride.

        Provided a list of regions [strand, left, right] (inclusive), steps across each region with
        steps of length stride and divides it into sub-regions of a given window length. If there is
        a remainder after splitting (an additional complete window can't fit), it will be split on
        the left and right sides of the input region.
        
        Parameters:
        ----------
        region_array : numpy array, shape (n_regions, 3)
            List of regions, inclusive, with columns of strand, left genomic position, and right
            genomic postion.

        window_length : int
            Length of the windows for regions to be subdivided into.
        
        stride : int
            Length of the steps for subdividing regions. For example, if stride < window_length,
            windows will be overlapping.

        Returns:
        ----------
        region_output : numpy array, shape (n_regions, 3)
            List of regions as above region_array, but the subdivided according to window_length and
            stride.
    """
    region_output = np.empty((0,3), dtype=np.int)
    for region in region_array:
        strand, left, right = region
        region_length = right - left + 1
        if region_length >= window_len:
            new_lefts, new_rights = _splitregion(region_length, window_len, stride)
            new_regions = np.asarray([np.zeros(len(new_lefts))+strand, new_lefts+left, new_rights+left], dtype=np.uint32).T
            region_output = np.r_[region_output, new_regions]
    return region_output
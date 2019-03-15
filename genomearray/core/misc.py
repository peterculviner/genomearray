import numpy as np

def concatregions(in_regions):
    """ Combines overlapping regions in a list of input regions into single regions.

        Iterates across the in_regions list and combines all regions with any overlap into single
        regions. Group of overlapping regions are combined into a single region even if the first
        member of the group is not overlapping the last member of the group.
        
        Parameters:
        ----------
        in_regions : array-like, shape (n regions, 3)
            The first column is presumed to be strand (0 or 1), the second column defines the left
            genomic position of the region (inclusive), and the third column defines the right
            genomic position of the region (inclusive).

        Returns:
        ----------
        out_regions : numpy array, shape (n regions, 3)
            Outputs numpy array in same form as in_regions but with overlapping regions combined
            into single regions.
    """
    out_regions = []
    if len(in_regions) == 0: # if no regions, then return empty
        return np.asarray(out_regions)
    for strand in [0,1]:
        # sort by left position
        on_strand = in_regions[in_regions[:,0] == strand]
        if len(on_strand) <= 1: # if 0 or 1 on this strand, then skip
            continue
        in_stack = list(on_strand[np.argsort(on_strand[:,1])[::-1]])
        out_regions.append(in_stack.pop())
        while len(in_stack) > 0:
            next_value = in_stack.pop()
            _, next_left, next_right = next_value
            _, _, curr_right = out_regions[-1]
            if (next_left <= curr_right):
                out_regions[-1][2] = max(next_right,curr_right)
            else:
                out_regions.append(next_value)
    return np.asarray(out_regions)

def regionstomask(in_regions, genome_len):
    """ Makes a genome-shaped (2, genome_len) True / False mask based on in_regions.

        All positions which fall into in_regions will be marked as True.
        
        Parameters:
        ----------
        in_regions : array-like, shape (n regions, 3)
            The first column is presumed to be strand (0 or 1), the second column defines the left
            genomic position of the region (inclusive), and the third column defines the right
            genomic position of the region (inclusive).

        genome_len : int
            Length of the genome must be provided to make the returned mask have the proper shape.

        Returns:
        ----------
        out_mask : numpy array, shape (2, genome_len)
            A mask of shape (2, genome_len) with positions included in in_regions as True.
    """
    out_mask = np.zeros((2,genome_len)).astype(bool)
    for region in in_regions:
        out_mask[region[0],region[1]:region[2]+1] = True
    return out_mask

def masktoregions(in_mask):
    """ Finds contiguous True regions on a genome-shaped mask.

        Accepts a genome-shaped boolean mask and identifies regions which are contiguous True.
        Returns regions [[strand, start, end]....] (inclusive) which are contiguous True.

        
        Parameters:
        ----------
        in_mask : numpy array, boolean shape (2, len(genome))
            Function finds contiguous True regions on this mask.

        Returns:
        ----------
        out_regions : numpy array, shape (n regions, 3)
            An array of regions which are contiguously True. The first column is the strand
            (0 or 1), the second column defines the left genomic position of the region (inclusive),
            and the third column defines the right genomic position of the region (inclusive).
    """
    regions = []
    for i in [0,1]: # do the thing for the first and second strands
        current_strand = in_mask[i].copy().astype(float)
        current_strand[-1] = np.nan # set final position to np.nan to avoid overlap issues
        transitions = current_strand - np.roll(current_strand,1)
        true_start = np.where(transitions == 1)[0]
        true_end   = np.where(transitions == -1)[0] - 1
        if current_strand[0] == 1: # if starts on True, add True start to front end
            true_start = np.r_[0,true_start]
        if in_mask[i][-1] == True: # if ends on True, add True end to back end
            true_end = np.r_[true_end, len(current_strand)-1]
            if in_mask[i][-2] == False: # if the one before is False, it's a single point True
                true_start = np.r_[true_start,len(current_strand)-1]
        if np.all(in_mask[i][-2:] == [True, False]):
            true_end = np.r_[true_end, len(current_strand)-2]
        regions.append(np.asarray([np.zeros(len(true_start))+i,true_start,true_end]).T)
    out_regions = np.concatenate(regions,axis=0).astype(int)
    return out_regions

def argoverlappingregions(input_region, region_array):
    """ Returns indexes over regions overlapping with the input region from region_array.
        
        Parameters:
        ----------
        input_region : array-like, shape (3,) order is strand, left, right (inclusive)
            Region to compare against the region_array.

        region_array : array-like, shape (n regions, 3)
            The first column is presumed to be strand (0 or 1), the second column defines the left
            genomic position of the region (inclusive), and the third column defines the right
            genomic position of the region (inclusive).

        Returns:
        ----------
        overlap_i : numpy array of int
            The indexes of regions in region_array which overlap with input_region.
    """
    overlap_i = np.where(np.all([input_region[0] == region_array[:,0],
                                  input_region[1] <= region_array[:,2],
                                  input_region[2] >= region_array[:,1]],axis=0))
    return overlap_i[0]

def subtractregion(new_region, old_regions):
    # find any overlapping regions
    overlap_i = np.where(np.all([new_region[0] == old_regions[:,0],
                                 new_region[1] <= old_regions[:,2],
                                 new_region[2] >= old_regions[:,1]],axis=0))[0]
    try: # try to delete first overlap member and recurse
        updated_regions = np.delete(old_regions,overlap_i[0],axis=0)
        deleted_region = old_regions[overlap_i[0]]
        # now resolve overlap, four possible cases
        if new_region[1] <= deleted_region[1] and new_region[2] >= deleted_region[2]: # new region completely ablates old region
            split_regions = np.empty((0,3)).astype(int)
        elif new_region[1] > deleted_region[1] and new_region[2] < deleted_region[2]: # new region is inside old region
            split_regions = [[deleted_region[0],deleted_region[1],new_region[1]-1], # left split
                             [deleted_region[0],new_region[2]+1,deleted_region[2]]] # right split
        elif new_region[1] <= deleted_region[1]: # new region is left overlapping
            split_regions = [[deleted_region[0],new_region[2]+1,deleted_region[2]]]
        elif new_region[2] >= deleted_region[2]: # new region is right overlapping
            split_regions = [[deleted_region[0],deleted_region[1],new_region[1]-1]]
        else:
            raise ValueError('unhandled region overlap type.')
        return subtractregion(new_region,np.r_[updated_regions,split_regions])
    except IndexError: # unless no overlapping regions left, then return old_regions
        return old_regions
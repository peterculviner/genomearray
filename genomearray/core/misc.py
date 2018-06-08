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
    for strand in [0,1]:
        # sort by left position
        on_strand = in_regions[in_regions[:,0] == strand]
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

def regionmask(in_regions, genome_len):
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
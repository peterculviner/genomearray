
def getFunctionOnRegions(function, regions, input_array):
    """ Conducts the given function on the given regions [[strand, left_position, right_position]....] of the input_array. """
    output = np.zeros(len(regions)) + np.nan
    for i,r in enumerate(regions):
        try: output[i] = function(input_array[r[0],r[1]:r[2]+1])
        except: pass # if function returns an exception, keep region output as np.nan
    return output

def getFunctionOnPositions(function, positions, input_array, addl_nt=0):
    """ Conducts the given function on regions or positions with additional buffer nucleotide positions on either side. """
    if positions.shape[1] == 2: # is a position record [[strand, position] ....]
        return getFunctionOnRegions(function, 
                                    np.asarray([positions[:,0], positions[:,1]-addl_nt, positions[:,1]+addl_nt]).T,
                                    input_array)
    elif positions.shape[1] == 3: # is a region record [[strand, left_position, right_position] .... ]
        return getFunctionOnRegions(function,
                                    np.asarray([positions[:,0], positions[:,1]-addl_nt, positions[:,2]+addl_nt]).T,
                                    input_array)

def sliceGenomeArray(input_array, strand, left, right):
    """Return 5'->3' slice of genome array based on inclusive coordinantes."""
    if strand == 0:
        return input_array[strand,left:right+1]
    elif strand == 1:
        return np.flip(input_array[strand,left:right+1],axis=0)
    else:
        raise ValueError('Invalid strand, choose 0 for + and 1 for -.')
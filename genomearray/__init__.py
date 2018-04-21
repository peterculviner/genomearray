import numpy as np

def getFunctionOnRegions(function, regions, input_array):
    """ Conducts the given function on the given regions (strand, start, end) of the input_array. """
    output = np.zeros(len(regions)) + np.nan
    for i,r in enumerate(regions):
        try: output[i] = function(input_array[r[0],r[1]:r[2]+1])
        except: pass # if function returns an exception, keep region output as np.nan
    return output

def getFunctionOnPositions(function, positions, input_array, addl_nt=0):
    if positions.shape[1] == 2: # is a vanilla position record
        return getFunctionOnRegions(function, 
                                    np.asarray([positions[:,0], positions[:,1]-addl_nt, positions[:,1]+addl_nt]).T,
                                    input_array)
    elif positions.shape[1] == 3: # actually is a region record
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

def getOneHotRepresentation(string,dtype=np.uint8):
    """ A,T,G,C 
        0 1 2 4 """
    string_representation = []
    # if string is empty, return an empty array of shape (0,4)
    if len(string) == 0:
        return np.empty(shape=(0,4))
    for character in string:
        if character == 'A':
            string_representation.append([1,0,0,0])
        elif character == 'T':
            string_representation.append([0,1,0,0])
        elif character == 'G':
            string_representation.append([0,0,1,0])
        elif character == 'C':
            string_representation.append([0,0,0,1])
        else:
            raise ValueError('Unexpected nucleotide character!')
    return np.asarray(string_representation,dtype=dtype)

def getGenomeRepresentation(genome, additional_data_arrays):
    if len(additional_data_arrays) != 0:
        additional_data_arrays = np.asarray(additional_data_arrays)
        genome_fwd = np.concatenate([getOneHotRepresentation(str(genome.seq)),
                                     additional_data_arrays[:,0,:].T],axis=1)
        genome_rev = np.concatenate([np.flip(getOneHotRepresentation(str(genome.seq.reverse_complement())),0),
                                     additional_data_arrays[:,1,:].T],axis=1)
    else:
        genome_fwd = getOneHotRepresentation(str(genome.seq))
        genome_rev = np.flip(getOneHotRepresentation(str(genome.seq.reverse_complement())),0)
    genome_onehot = np.asarray([genome_fwd, genome_rev])
    return genome_onehot

def convolveGenome(genome_representation, pwm):
    score_fwd = np.reshape(convolve(np.flip(np.flip(pwm.T,0),1),genome_representation[0],mode='valid'),-1)
    score_fwd = np.r_[score_fwd, np.zeros(genome_representation.shape[1] - score_fwd.shape[0]).astype(int)]
    score_rev = np.reshape(convolve(np.flip(np.flip(pwm.T,0),1),np.flip(genome_representation[1],0),mode='valid'),-1)
    score_rev = np.flip(np.r_[score_rev, np.zeros(genome_representation.shape[1] - score_rev.shape[0])],0)
    return np.asarray([score_fwd, score_rev])

def getPositionWeightMatrix(freq_array, background_freq_array):
    """Generates a position weight matrix scoring table (rows = <A,T,G,C>, columns = position) using
       a given base frequency array and a background base frequency array."""
    pwm_scoring_matrix = np.log2(freq_array / background_freq_array)
    return pwm_scoring_matrix
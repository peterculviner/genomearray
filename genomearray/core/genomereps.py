import numpy as np

def dnatoonehot(string,dtype=np.uint8):
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

def addChannels(genome_representation, additional_arrays):
    """ Add additional channels to genome representation from list of additional arrays. """
    additional_arrays = np.asarray(additional_arrays)
    concatenated_representation = np.concatenate([genome_representation,np.reshape(additional_data_arrays[:,:,:],(additional_data_arrays.shape[1],additional_data_arrays.shape[0],-1))])
    return concatenated_representation

def genometoonehot(genbank_file):
    genome_fwd = dnatoonehot(str(genbank_file.seq))
    genome_rev = np.flip(dnatoonehot(str(genbank_file.seq.reverse_complement())),0)
    genome_onehot = np.asarray([genome_fwd, genome_rev])
    return genome_onehot

def extractntonehot(onehot_genome, positions, five, three):
    extracted_nt = []
    for s,pos in positions:
        if s == 0:
            extracted_nt.append(onehot_genome[s,pos-five:pos+three])
        elif s == 1:
            extracted_nt.append(np.flip(onehot_genome[s,pos-three+1:pos+five+1],0))
    return extracted_nt



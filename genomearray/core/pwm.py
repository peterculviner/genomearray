import numpy as np
from scipy.signal import convolve

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
import numpy as np
from scipy.signal import convolve

def getGenomeConvolution(genome_representation, pwm):
    """ Returns the convolution of a position weight matrix (shape is (pwm_position, nt_positions))
        across genome representation in a 5'-> 3' direction.

        PWM should be provided in shape of (pwm_position, nt_position). Genome should be provided 
        in shape of (strand, nt_position, pwm_position).
        
        Parameters:
        ----------
        genome_representation : numpy array
            TO DO

        pwm : numpy array
            TO DO

        Returns:
        ----------
        out : numpy array
            Returns zero-padded numpy array of same shape as genome representation.
        """
    score_fwd = np.reshape(convolve(np.flip(np.flip(pwm.T,0),1),genome_representation[0],mode='valid'),-1)
    score_fwd = np.r_[score_fwd, np.zeros(genome_representation.shape[1] - score_fwd.shape[0]).astype(int)]
    score_rev = np.reshape(convolve(np.flip(np.flip(pwm.T,0),1),np.flip(genome_representation[1],0),mode='valid'),-1)
    score_rev = np.flip(np.r_[score_rev, np.zeros(genome_representation.shape[1] - score_rev.shape[0])],0)
    out = np.asarray([score_fwd, score_rev])
    return out

def getPositionWeightMatrix(freq_array, background_freq_array):
    """Generates a position weight matrix scoring table (rows = <A,T,G,C>, columns = position) using
       a given base frequency array and a background base frequency array."""
    pwm_scoring_matrix = np.log2(freq_array / background_freq_array)
    return pwm_scoring_matrix
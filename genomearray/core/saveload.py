import pysam
import numpy as np
import genomearray as ga
from scipy.stats import gmean

def loadarrays(array_paths, normalization=None, **kwargs):
    """ Load arrays (.npy files) and conduct normalization across the datasets.

        Arrays are loaded with np.load, normalization functions can be provided using the
        normalization kwarg. Default is no normalization.
        
        Parameters:
        ----------
        array_paths : list of file paths to arrays to load
            Loads each of the array paths in turn using np.load.

        normalization : None (default) or a function to normalize arrays on loading
            Function for normalizing a list of arrays. 

        **kwargs : additional kwargs
            Passed to normalization function (if present) as kwargs.

        Returns:
        ----------
        out : array of normalized datasets
            
    """
    loaded_arrays = np.asarray([np.load(path) for path in array_paths])
    if normalization is None:
        return loaded_arrays
    else:
        return normalization(loaded_arrays, **kwargs)

def _mediansizefactors(samples, gene_regions):
    # axis 0 = samples; axis 1 = gene_sums
    sample_sums = np.asarray([ga.regionfunc(np.sum, gene_regions, s) for s in samples])+1
    # generate a reference sample to normalize to
    reference_sample = gmean(np.asarray(sample_sums), axis=0)
    # divide sample genes by reference samples
    gene_ratios = sample_sums / reference_sample.reshape(1,-1)
    size_factors = np.nanmedian(gene_ratios, axis=1)
    return size_factors

def countnormalization(sample_arrays, paths_to_bams = None, log2 = None):
    # calculate size factors from raw reads mapped to bam files
    counts = []
    for path in paths_to_bams:
        counts.append(pysam.Samfile(path, 'rb').mapped)
    counts = np.asarray(counts)
    size_factors = counts / gmean(counts)
    # now normalize the arrays
    normalized_sample_arrays = (sample_arrays + 1) / size_factors.reshape(-1,1,1)
    if log2:
        return np.log2(normalized_sample_arrays)
    elif log2 == False:
        return normalized_sample_arrays
    raise ValueError('log2 must be set to True or False.')

def mediandensitynormalization(sample_arrays, regions = None, log2 = None):
    size_factors = _mediansizefactors(sample_arrays, regions)
    normalized_sample_arrays = (sample_arrays + 1) / size_factors.reshape(-1,1,1)
    if log2:
        return np.log2(normalized_sample_arrays)
    elif log2 == False:
        return normalized_sample_arrays
    raise ValueError('log2 must be set to True or False.')
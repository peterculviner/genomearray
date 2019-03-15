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

def regionsumnormalization(sample_arrays, regions = None, log2 = None):
    sample_sums = []
    for counts in sample_arrays:
        sample_sums.append(np.sum(ga.regionfunc(np.sum, regions, counts)))
    size_factors = np.asarray(sample_sums) / gmean(sample_sums,axis=0)
    if log2:
        return (sample_arrays + 1) / size_factors.reshape(-1,1,1)
    elif log2 == False:
        return sample_arrays / size_factors.reshape(-1,1,1)
    raise ValueError('log2 must be set to True or False')

def mediandensitynormalization(sample_arrays, regions = None, log2 = None):
    size_factors = _mediansizefactors(sample_arrays, regions)
    normalized_sample_arrays = (sample_arrays + 1) / size_factors.reshape(-1,1,1)
    if log2:
        return np.log2(normalized_sample_arrays)
    elif log2 == False:
        return normalized_sample_arrays
    raise ValueError('log2 must be set to True or False.')

def loadarrays2d(array_paths, normalization=None, **kwargs):
    all_loaded_arrays = []
    for sample_arrays in array_paths:
        sample_loaded_arrays = []
        for single_array in sample_arrays:
            sample_loaded_arrays.append(np.load(single_array))
        all_loaded_arrays.append(sample_loaded_arrays)
    if normalization is None:
        return all_loaded_arrays
    else:
        return normalization(all_loaded_arrays, **kwargs)
    
def regionsumnormalization2d(sample_arrays2d, multi_regions = None, log2 = None):
    sample_sums = []
    for sample_arrays in sample_arrays2d:
        genome_sums = []
        for sample_counts, regions in zip(sample_arrays, multi_regions):
            genome_sums.append(np.sum(np.asarray(ga.regionfunc(np.sum, regions, sample_counts))))
        sample_sums.append(np.sum(genome_sums))
    size_factors = np.asarray(sample_sums) / gmean(sample_sums,axis=0)
    if log2:
        return [[np.log2((sample_counts+1)/factor) for sample_counts in sample]
                 for sample, factor in zip(sample_arrays2d,size_factors)]
    elif log2 == False:
        return [[(sample_counts)/factor for sample_counts in sample]
                 for sample, factor in zip(sample_arrays2d,size_factors)]
    raise ValueError('log2 must be set to True or False')
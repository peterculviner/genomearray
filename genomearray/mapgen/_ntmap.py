import numpy as np
import pysam

def mapfragdensity(path_to_bam, min_mapq=2, refseq_index=0, dtype=np.uint32):
    """ Given a paired-end, dUTP RNA-seq experiment, map fragment density at a single nt resolution.

        Function accepts an indexed bam file and uses pysam to iterate across all fragments. For
        each proper read pair, only one read is assessed and, if it meets the mapq threshold, a
        count is added to every nt position which the fragment crosses on the genome. A numpy array
        is returned with the counts. numpy data type of the output can be altered for RAM usage.
        
        Parameters:
        ----------
        path_to_bam : path to bam file (string)
            Must be an indexed bam file.

        min_mapq : mapping quality required for a read to be mapped (int), default 2
            For bowtie2, reads with mapq == 0 are equally likely to map to more than one location 
            in the reference sequences. To avoid all reads which are likely to multiply map, use
            a min_mapq of 2.

        refseq_index : index of reference sequence to generate numpy array of (int), default 0
            If multiple reference sequences were provided to bowtie for mapping, the reference to
            generate a fragment density for must be provided.

        dtype : numpy data type, default np.uint32
            Data type of output numpy array.

        Returns:
        ----------
        density_array : numpy array
            2 x reference sequence length array of fragment counts at each nt position.

    """
    def filterReads(reads):
        for read in reads: # yield only reads which are proper paired, meet mapq cutoff, and are one side of the fragment
            if read.is_proper_pair and read.mapping_quality >= min_mapq and not read.is_reverse:
                yield read
    # prepare the output array and load the experiment
    bam = pysam.Samfile(path_to_bam, "rb") # load the bam file
    density_array = np.zeros((2,bam.lengths[refseq_index]), dtype)
    # load and filter the reads and add them to the density array
    filtered_reads = filterReads(bam.fetch(bam.references[refseq_index]))
    for read in filtered_reads:
        if read.is_read1: # maps to the minus strand
            density_array[1,read.pos:read.pos+np.abs(read.template_length)] += 1
        else: # maps to the plus strand
            density_array[0,read.pos:read.pos+np.abs(read.template_length)] += 1
    return density_array
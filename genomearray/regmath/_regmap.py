import numpy as np
import pysam


def mapregioncounts(path_to_bam, regions, mapq_cutoff=2, refseq_index=0):
    # regions must be in form [[strand, left, right], ....]
    # define a filter to only consider half of reads (left-most read)
    def filterReads(reads):
        for read in reads: # yield only reads which are proper paired, meet mapq cutoff, and are one side of the fragment
            if read.is_proper_pair and read.mapping_quality >= mapq_cutoff and not read.is_reverse:
                yield read
    # do the magic....
    bam = pysam.Samfile(path_to_bam, "rb") # load the bam file
    out_counts = np.zeros(len(regions))
    aligned_sizes = [] # record aligned region size
    left_region_pointer = 0 # where in the reference the most recent read was
    regioned_fragments = 0 # fragments overlapping with regions
    unregioned_fragments = 0 # fragments not overlapping with regions
    filtered_reads = filterReads(bam.fetch(bam.references[refseq_index])) # get reads defined by filter function above
    for read in filtered_reads:
        aligned_sizes.append(read.template_length) # record aligned size to template
        # determine strand
        if read.is_read1:
            strand = 1 # negative strand
        else:
            strand = 0 # positive strand
        left = read.pos
        right = read.pos+np.abs(read.template_length)-1
        found_region = False # flag to determine if read ever mapped to a region
        ## see if we should advance the region pointer
        # if left position of read is greater than right edge of read, we should stop considering these regions
        try:
            while regions[left_region_pointer,2] < left: 
                left_region_pointer += 1 # stop looking at region, its time has passed....
             # now try to compare the read to the remaining regions to check for overlap
            right_region_pointer = left_region_pointer
            while regions[right_region_pointer,1] <= right:
                if (strand == regions[right_region_pointer,0] and
                        right >= regions[right_region_pointer,1] and
                        left <= regions[right_region_pointer,2]): # check for overlap, if true, add a count
                    out_counts[right_region_pointer] += 1
                    found_region = True
                right_region_pointer += 1
            if found_region:
                regioned_fragments+=1
            else:
                unregioned_fragments+=1
        except IndexError: # index error here indicates we've passed last region, all reads are now unregioned
            unregioned_fragments += 1
    return out_counts, regioned_fragments, unregioned_fragments, aligned_sizes
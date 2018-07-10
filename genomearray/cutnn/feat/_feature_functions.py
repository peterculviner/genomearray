import numpy as np
import genomearray as ga

def buildbinaryfeatures(positivefunc, positivekwargs,
                        negativefunc, negativekwargs,
                        negsamplerfunc, negsamplerkwargs,
                        featuregen,  featuregenkwargs,
                        verbose=False, random_seed=None):
    # to enable reproducibility, allow insert of a random-seed
    np.random.seed(random_seed)
    # get locations of events using the positive feature function
    positive_events  = positivefunc(**positivekwargs)
    if verbose == True:
        print 'found %i positive events.' % (positive_events.shape[0])
    # get possible negative regions using the given negative feature function
    negative_regions = negativefunc(**negativekwargs)
    if verbose == True:
        print '%.2f percent of genome flagged as negative.' % (float(np.sum(negative_regions == True)) / np.sum(negative_regions == False) * 100)
    # input negative_regions and positive_events into the negative sampler function
    negative_events  = negsamplerfunc(negative_regions, positive_events, **negsamplerkwargs)
    if verbose == True:
        print '...converted into %i negative events.' % negative_events.shape[0]
    # make features from provided positions
    positive_features = featuregen(positive_events, **featuregenkwargs)
    negative_features = featuregen(negative_events, **featuregenkwargs)
    # combine features, generate labels and weights
    features = [np.concatenate([pfeat,nfeat],0) for pfeat,nfeat in zip(positive_features,negative_features)]
    feature_labels = np.r_[np.asarray([np.zeros(len(positive_features[0])),np.ones(len(positive_features[0]))]).T,
                           np.asarray([np.ones(len(negative_features[0])),np.zeros(len(negative_features[0]))]).T].astype(np.int8)
    # for feature weights, set positive samples to 1, negative samples to nessasary value for equality
    feature_weights = np.r_[np.ones(len(positive_features[0])),
                            np.zeros(len(negative_features[0]))+
                            len(positive_features[0])/float(len(negative_features[0]))].astype(np.float16)
    # generate a shuffle order
    shuffle_order = np.random.choice(np.arange(len(feature_weights)), size=len(feature_weights), replace=False)
    feature_out = [features, feature_labels, feature_weights]
    return feature_out, shuffle_order, positive_events, negative_events

def randomregionsampler(negative_mask, positive_positions, n_samples=None, buffer_size=None):
    negative_mask = negative_mask.copy()
    # subtract all regions generated from positive_positions +/- buffer_size
    for positive in positive_positions:
        if positive.shape == (2,):
            strand, pos = positive
            negative_mask[strand,pos-buffer_size:pos+buffer_size+1] = False
        elif positive.shape == (3,):
            strand, left, right = positive
            negative_mask[strand,left-buffer_size:right+buffer_size+1] = False
        else:
            raise ValueError('Unhandled positive event type.')
    # now convert the negative mask into region for negative sampling
    regions = ga.masktoregions(negative_mask)
    # now sample until number of required samples is met
    sampled = 0
    sample_positions = []
    while n_samples != sampled:
        # get the lengths of all regions
        region_lengths = regions[:,2] - regions[:,1] + 1
        # assign probability of picking region by its size less 2*buffer_size
        len_minus_buffer = np.maximum(0,region_lengths - 2*buffer_size) # don't let any sizes be below 0
        if np.sum(len_minus_buffer) > 0:
            # pick a region based on how many possible samples there are in it
            probabilities = len_minus_buffer/np.sum(len_minus_buffer).astype(float)
            region_i = np.random.choice(np.arange(len(regions)),p=probabilities) # pick a region
            strand, left, right = regions[region_i] # get position information for picked region
            base_i = np.random.choice(np.arange(len_minus_buffer[region_i])) # pick a position in the region
            sample_positions.append([strand,left+buffer_size+base_i]) # add the sample position
            sampled_region = [strand,left+base_i,left+base_i+2*buffer_size]
        else:
            break # break the while loop if there are no possible samples left
        regions = ga.subtractregion(sampled_region, regions)
        sampled += 1
    return np.asarray(sample_positions)

def ntfeatures(positions, regions=None, genome=None,
                          array_types=None, offset_terms=None,
                          additional_data_arrays=[], output_positions=False):
    # generate out arrays for each of the messages
    out_arrays = [[] for i in range(len(array_types))]
    included_positions = []
    # iterate across all positions
    for s_p in positions:
        strand, pos = s_p
        # get all region(s) which overlap with the position of interest
        overlapping_regions = regions[ga.argoverlappingregions([strand, pos, pos], regions)]
        if overlapping_regions.shape[0] != 1:
            continue # only record if position can be unambiguously assigned to a single region
        included_positions.append(s_p)
        # get region information
        r_strand, r_start, r_end = overlapping_regions[0]
        # generate output arrays for each of the given array types
        for i, a_type, a_offset in zip(range(len(array_types)), array_types, offset_terms):
            if a_type == 'five':
                if strand == 0: # position & region fall on positive strand
                    left, right = (r_start, pos+a_offset+1)
                    genome_string = str(genome.seq[left:right])
                elif strand == 1: # position & region fall on negative strand
                    left, right = (pos-a_offset, r_end+1)
                    genome_string = str(genome.seq[left:right].reverse_complement())
            elif a_type == 'three':
                if strand == 0: # position & region fall on positive strand
                    left, right = (pos+a_offset, r_end+1)
                    genome_string = str(genome.seq[left:right])
                elif strand == 1: # position & region fall on negative strand
                    left, right = (r_start,pos-a_offset+1)
                    genome_string = str(genome.seq[left:right].reverse_complement())
            elif a_type == 'centered':
                if strand == 0: # position & region fall on positive strand
                    left, right = (pos+a_offset[0], pos+a_offset[1]+1)
                    genome_string = str(genome.seq[left:right])
                elif strand == 1: # position & region fall on negative strand
                    left, right = (pos-a_offset[1], pos-a_offset[0]+1)
                    genome_string = str(genome.seq[left:right].reverse_complement())
            else:
                raise ValueError('Unsupported array type.')
            # now that left / right have been defined, add the feature to the output array
            string_rep = ga.dnatoonehot(genome_string)
            concat_input = [string_rep] # prepare inputs for concatenation
            for a in additional_data_arrays:
                concat_input.append(np.reshape(ga.genomeslice(a, r_strand, left, right-1),(-1,1)))
            out_arrays[i].append(np.concatenate(concat_input, axis=1))
    if output_positions:
        return out_arrays, np.asarray(included_positions)
    else:
        return out_arrays

def targetregionfeatures(target_region, sampling_step, ntfeatures_kwargs):
    # get positions for sampling
    strand, left, right = target_region
    sample_bp_positions = np.arange(left, right, sampling_step)
    sample_positions = np.asarray([np.zeros(sample_bp_positions.shape[0])+strand, sample_bp_positions]).astype(int).T
    ntfeatures_kwargs = ntfeatures_kwargs.copy()
    ntfeatures_kwargs['output_positions'] = True
    sample_features, sample_positions = ga.cutnn.feat.ntfeatures(sample_positions, **ntfeatures_kwargs)
    return sample_positions, sample_features

def regionlistfeatures(region_list, sampling_step, ntfeatures_kwargs):
    # do target region
    positions_list = []
    features_list = []
    for target_region in region_list:
        positions, features = ga.cutnn.feat.targetregionfeatures(target_region, sampling_step, ntfeatures_kwargs)
        positions_list.append(positions)
        features_list.append(features)
    sample_positions = np.concatenate(positions_list,0)
    sample_features = []
    for i in range(len(features_list[0])):
        sample_features.append(np.concatenate([f[i] for f in features_list],0))
    return sample_positions, sample_features

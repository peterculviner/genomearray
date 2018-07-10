import numpy as np

def randshuffle(shuffle_order, features):
    """ Return a new shuffle order without altering or looking at features. """
    return np.random.choice(shuffle_order, size=len(shuffle_order), replace=False)

def datasplitter(input_i, data_split):
    if callable(data_split): # if data split is a function
        train_i, validate_i, test_i = data_split(shuffle_order, features, **split_parameters)
    elif np.sum(data_split) == 10 and len(data_split) == 3:
        split_i = np.array_split(input_i, 10) # split data into 10 ~equal parts
        train_i, validate_i, test_i = (np.concatenate(split_i[:data_split[0]]),
                                       np.concatenate(split_i[data_split[0]:data_split[0]+data_split[1]]),
                                       np.concatenate(split_i[data_split[0]+data_split[1]:]))
        return train_i, validate_i, test_i
    else:
        raise ValueError('fractional split must sum to 10 and be divided into (training, validation, testing)')

def batchiter(features, shuffle_order, batch_size=100, mode=None,
              data_split=(6,2,2), # either a function or a list of 3 numbers summing to 10
              split_parameters={}, # additional kwargs to pass to split function, if present
              shuffle_function=lambda shuffle_order, features: shuffle_order, # placeholder function which does not shuffle
              shuffle_parameters={}, # additional kwargs to pass to shuffle function
              data_modification=lambda x:x, # placeholder function with no modification to data
              single_cycle = False,
              ):
    """ For data handling, choose from train, test or validation.
    Features must be in the form of a list: [x, labels, sample_weights]."""
    # split the features using the shuffle order
    train_i, validate_i, test_i = datasplitter(shuffle_order, data_split)
    while True: # data yielding loop to run until generator is no longer needed
        # reshuffle the data indices in the desired set gathered from the split function
        if mode == 'train':
            input_order = shuffle_function(train_i,    features, **shuffle_parameters)
        elif mode == 'validate':
            input_order = shuffle_function(validate_i, features, **shuffle_parameters)
        elif mode == 'test':
            input_order = shuffle_function(test_i,     features, **shuffle_parameters)
        else:
            raise ValueError('Choose from train, test, or validation for mode.')
        # internal for loop generates batch sized pieces of single epochs of data with shuffling (above) in between
        for batch_start in range(len(input_order)/batch_size):
            batch_i = input_order[batch_start*batch_size:batch_start*batch_size+batch_size]
            x = [] # for storage of current x (input features)
            for input_i in range(len(features[0])): # for multiple input NNs, iterate across inputs
                # append this batch (batch_i) of this input (input_i) of features (features[0])
                # data is also run through a user-defined modification function
                x.append(data_modification(features[0][input_i][batch_i]))
            y = features[1][batch_i] # data labels can accessed directly
            w = features[2][batch_i] # sample weights can be accessed directly
            yield x, y, w # output batch
        if single_cycle: # break the while loop if only a single cycle through the data is desired
            break
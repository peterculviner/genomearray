import numpy as np
import genomearray as ga

def mappredictions(model, positions, features, genome_length):
    # use the model and features to generate predictions
    predictions = model.predict(np.asarray(features[0]))
    positive_rate = predictions[:,1]
    # generate a data map for the genome
    prediction_map = np.zeros((2,genome_length)) + np.nan
    prediction_map[tuple(np.asarray(positions).T)] = positive_rate
    return prediction_map

def rawpredictions(model, features):
    # use the model and features to generate predictions
    predictions = model.predict(np.asarray(features[0]))
    positive_rate = predictions[:,1]
    return positive_rate
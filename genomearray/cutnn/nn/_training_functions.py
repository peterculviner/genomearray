import genomearray as ga
from sklearn.metrics import roc_auc_score


def fitmodel(model, features, shuffle_order, batch_size=100, max_epochs=100, data_split=(6,2,2),
             keras_callbacks=None, verbose=True, save_path=None):
    # make generators
    trainiter = ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                          batch_size=batch_size, mode='train', shuffle_function=ga.cutnn.nn.randshuffle)
    validiter = ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                          batch_size=batch_size, mode='validate')
    testiter  = ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                          batch_size=batch_size, mode='test')
    # get step counts using the additional single_cycle arg
    train_steps    = len(list(ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                                              batch_size=batch_size, mode='train', single_cycle = True)))
    validate_steps = len(list(ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                                              batch_size=batch_size, mode='validate', single_cycle = True)))
    test_steps     = len(list(ga.cutnn.nn.batchiter(features, shuffle_order, data_split=data_split,
                                              batch_size=batch_size, mode='test', single_cycle = True)))
    # fit the model
    model.fit_generator(trainiter, train_steps, epochs=max_epochs, verbose=verbose,
                        validation_data=validiter, validation_steps=validate_steps,
                        callbacks=keras_callbacks)
    # now load best model weights and run test data on it
    model.load_weights(save_path)
    y_predict = model.predict_generator(testiter, test_steps)
    _, _, test_i = ga.cutnn.nn.datasplitter(shuffle_order, data_split)
    y_true = features[1][test_i][:y_predict.shape[0]]
    return roc_auc_score(y_true, y_predict)
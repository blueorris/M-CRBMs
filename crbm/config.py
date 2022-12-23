HIDDEN_UNITS = 8000 # number of nodes in hidden layer
BATCH_SIZE = 64 # batch size for both training data and test data
EPOCHS = 1 # training epochs
LEARNING_RATE = 1e-3 # learning rate of the Adam optimizer (ignore this if use different optimizer)
WEIGHT_DECAY = 1e-3 # weight decay of the Adam optimizer (ignore this if use different optimizer)
CD_K = 1 # number of steps of Gibbs sampling in contrastive divergence, default is 1
OPTIM = 'rms' # Set optimizer, 'rms' or 'adam'
CATEGORY = 'artist_genre_series' # What categories will be used, this will be shown in trained model's file name
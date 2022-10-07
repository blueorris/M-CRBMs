import tensorflow as tf
from cond_rbm import CondRBM

tf.config.set_soft_device_placement(True)


def _parse_function(array):
    split = tf.strings.to_number(tf.compat.v1.string_split([array], " ").values, out_type=tf.dtypes.int64)
    indices = tf.reshape(split[1:][::2], [-1, 1])
    values = tf.add(split[1:][1::2], -1)
    dense_shape = [180508]

    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


dset = tf.data.TextLineDataset("./visible_training.csv")
dataset = dset.map(_parse_function)

r1_raw = tf.data.TextLineDataset("./r1_training.csv")
r1 = r1_raw.map(_parse_function)

r2_raw = tf.data.TextLineDataset("./r2_training.csv")
r2 = r2_raw.map(_parse_function)

r3_raw = tf.data.TextLineDataset("./r3_training.csv")
r3 = r3_raw.map(_parse_function)

dataset = tf.data.Dataset.zip((dataset, r1, r2, r3))

# prepare probe dataset
probe = tf.data.TextLineDataset("./visible_mytest.csv")
probe = dset.map(_parse_function)

r1_raw = tf.data.TextLineDataset("./r1_mytest.csv")
test_r1 = r1_raw.map(_parse_function)

r2_raw = tf.data.TextLineDataset("./r2_mytest.csv")
test_r2 = r2_raw.map(_parse_function)

r3_raw = tf.data.TextLineDataset("./r3_mytest.csv")
test_r3 = r3_raw.map(_parse_function)

probe_dataset = tf.data.Dataset.zip((probe, test_r1, test_r2, test_r3))

# call function

rbm = CondRBM()
with tf.device("/cpu:0"):
    rbm.train(dataset, 30, probe, probe_dataset)


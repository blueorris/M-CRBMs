import tensorflow as tf
import numpy as np
import datetime
from numpy import asarray
from numpy import save


class CondRBM:
    def __init__(self):
        self.n_visible = 180508
        self.batch_size = 100
        self.n_hidden = 100
        self.momentum = tf.constant(0.9)
        self.weight_decay = tf.constant(0.001)
        self.k = 1
        self.num_rat = 5
        self.start_time = datetime.datetime.now()
        self.lr_weights = tf.constant(0.0015)
        self.lr_vb = tf.constant(0.0012)
        self.lr_hb = tf.constant(0.1)
        self.lr_D = tf.constant(0.0005)

        self.anneal = False
        self.anneal_val = 0.0

        self.weights = tf.Variable(tf.random.normal([self.n_visible, self.num_rat, self.n_hidden], stddev=0.01),
                                   name='weights')
        self.hidden_bias = tf.Variable(tf.constant(0.0, shape=[self.n_hidden]), name='h_bias')
        self.visible_bias = tf.Variable(tf.constant(0.0, shape=[self.n_visible]), name='v_bias')

        self.weight_v = tf.zeros(tf.shape(self.weights))
        self.visible_bias_v = tf.zeros(tf.shape(self.visible_bias))
        self.hidden_bias_v = tf.zeros(tf.shape(self.hidden_bias))

        self.D_1 = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), name='D1')
        self.D_v_1 = tf.zeros(tf.shape(self.D_1))

        self.D_2 = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), name='D2')
        self.D_v_2 = tf.zeros(tf.shape(self.D_2))

        self.D_3 = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), name='D3')
        self.D_v_3 = tf.zeros(tf.shape(self.D_3))

    def forward_prop(self, visible, r1, r2, r3):
        '''Computes a vector of probabilities hidden units are set to 1 given visible unit states'''

        pre_activation = tf.add(tf.tensordot(visible, self.weights, [[1, 2], [1, 0]]), self.hidden_bias)

        r_D_1 = tf.tensordot(r1, self.D_1, [[1], [0]])
        r_D_2 = tf.tensordot(r2, self.D_2, [[1], [0]])
        r_D_3 = tf.tensordot(r3, self.D_3, [[1], [0]])

        add_r = tf.add_n([pre_activation, r_D_1, r_D_2, r_D_3])
        return tf.nn.sigmoid(add_r)

    def backward_prop(self, hidden):
        '''Computes a vector of probabilities visible units are set to 1 given hidden unit states'''

        return tf.nn.softmax(tf.add(tf.transpose(tf.tensordot(hidden, self.weights, [[1], [2]]), perm=[0, 2, 1]),
                                    self.visible_bias), axis=1)

    def sample_h_given_v(self, visible_sample, r1, r2, r3, binary=True):
        '''Sample a hidden unit vector based on state probabilities given visible unit states.'''

        hidden_probs = self.forward_prop(visible_sample, r1, r2, r3)
        if binary:
        # Relu is just to convert -1 to 0 for sampled vector
            return tf.nn.relu(tf.sign(tf.subtract(hidden_probs, tf.random.uniform(tf.shape(hidden_probs)))))
        else:
            return hidden_probs

    def sample_v_given_h(self, hidden_sample, binary=False):
        '''Sample a visible unit vector based on state probabilities given hidden unit states.'''

        visible_probs = self.backward_prop(hidden_sample)
        # Relu is just to convert -1 to 0 for sampled vector
        if binary:
            return tf.nn.relu(tf.sign(tf.subtract(visible_probs, tf.random_uniform(tf.shape(visible_probs)))))
        else:
            return visible_probs

    def CD_k(self, visibles, r1, r2, r3, mask):
        '''Contrastive divergence with k steps of Gibbs Sampling.'''

        orig_hidden = self.sample_h_given_v(visibles, r1, r2, r3)
        # k steps of alternating parallel updates
        for i in range(self.k):
            if i == 0:
                hidden_samples = orig_hidden
            # mask：选取输入visibles当中大于0的rating，设其值为1
            visible_samples = self.sample_v_given_h(hidden_samples) * mask
            if i == self.k - 1:
                hidden_samples = self.sample_h_given_v(visible_samples, r1, r2, r3, binary=False)
            else:
                hidden_samples = self.sample_h_given_v(visible_samples, r1, r2, r3, binary=True)

        w_grad_pos = tf.einsum('ai,ajm->mji', orig_hidden, visibles)
        w_neg_grad = tf.einsum('ai,ajm->mji', hidden_samples, visible_samples)
        w_grad_tot = tf.subtract(w_grad_pos, w_neg_grad)

        # user_rated = (5 / (tf.maximum(tf.reduce_sum(mask, axis=(0, 1)), 1)))
        # w_grad_tot = tf.einsum('i,ijk->ijk', user_rated, w_grad_tot)

        hb_grad = tf.reduce_mean(tf.subtract(orig_hidden, hidden_samples), axis=0)

        vb_grad = tf.reduce_sum(tf.subtract(visibles, visible_samples), axis=[0, 1])
        # vb_grad = tf.einsum('i,ji->ji', user_rated, vb_grad)

        D_grad_1 = tf.einsum('bh,bm->mh', tf.subtract(orig_hidden, hidden_samples), r1)
        D_grad_2 = tf.einsum('bh,bm->mh', tf.subtract(orig_hidden, hidden_samples), r2)
        D_grad_3 = tf.einsum('bh,bm->mh', tf.subtract(orig_hidden, hidden_samples), r3)
        # D_grad = tf.einsum('i,ij->ij', user_rated, D_grad)

        return w_grad_tot, hb_grad, vb_grad, D_grad_1, D_grad_2, D_grad_3

    def learn(self, visibles, r1, r2, r3):

        reduced_visibles = tf.reduce_sum(visibles, axis=1)

        mask = tf.where(tf.equal(reduced_visibles, 0), tf.zeros_like(reduced_visibles), tf.ones_like(reduced_visibles))
        mask = tf.stack([mask, mask, mask, mask, mask], axis=1)

        weight_grad, hidden_bias_grad, visible_bias_grad, D_grad_1, D_grad_2, D_grad_3 = \
            self.CD_k(visibles, r1, r2, r3, mask)
        return [tf.negative(weight_grad), tf.negative(hidden_bias_grad), tf.negative(visible_bias_grad),
                tf.negative(D_grad_1), tf.negative(D_grad_2), tf.negative(D_grad_3)]

    def get_variables(self):
        return [self.weights, self.hidden_bias, self.visible_bias, self.D_1, self.D_2, self.D_3]

    def apply_gradients(self, grads):
        self.weight_v = tf.add(grads[0], tf.scalar_mul(self.momentum, self.weight_v))

        # weight_update -= tf.scalar_mul(self.weight_decay, self.weights)
        self.weights.assign_add(self.weights, tf.scalar_mul(self.lr_weights, self.weight_v))

        self.hidden_bias_v = tf.add(grads[1], tf.scalar_mul(self.momentum, self.hidden_bias_v))
        self.hidden_bias.assign_add(self.hidden_bias, tf.scalar_mul(self.lr_hb, self.hidden_bias_v))

        self.visible_bias_v = tf.add(grads[2], tf.scalar_mul(self.momentum, self.visible_bias_v))
        self.visible_bias.assign_add(self.visible_bias, tf.scalar_mul(self.lr_vb, self.visible_bias_v))

        self.D_v_1 = tf.add(grads[3], tf.scalar_mul(self.momentum, self.D_v_1))
        self.D_1.assign_add(self.D_1, tf.scalar_mul(self.lr_D, self.D_v_1))

        self.D_v_2 = tf.add(grads[4], tf.scalar_mul(self.momentum, self.D_v_2))
        self.D_2.assign_add(self.D_2, tf.scalar_mul(self.lr_D, self.D_v_2))

        self.D_v_3 = tf.add(grads[5], tf.scalar_mul(self.momentum, self.D_v_3))
        self.D_3.assign_add(self.D_3, tf.scalar_mul(self.lr_D, self.D_v_3))

    def get_rx(self, iterator):
        training_point, r1, r2, r3 = iterator.get_next()

        r1_sparse = tf.SparseTensor(indices=r1.indices, values=tf.ones(tf.shape(r1.indices)[0]),
                                    dense_shape=[tf.shape(training_point)[0], self.n_visible])
        r2_sparse = tf.SparseTensor(indices=r2.indices, values=tf.ones(tf.shape(r2.indices)[0]),
                                    dense_shape=[tf.shape(training_point)[0], self.n_visible])
        r3_sparse = tf.SparseTensor(indices=r3.indices, values=tf.ones(tf.shape(r3.indices)[0]),
                                    dense_shape=[tf.shape(training_point)[0], self.n_visible])

        r1 = tf.sparse.to_dense(r1_sparse, validate_indices=False)
        r2 = tf.sparse.to_dense(r2_sparse, validate_indices=False)
        r3 = tf.sparse.to_dense(r3_sparse, validate_indices=False)
        x = tf.sparse.to_dense(training_point, default_value=-1)

        return x, r1, r2, r3

    def get_save_path_name(self):
        st = self.start_time.strftime('%y%m%d')
        return "./temp/cond_rbm_" + st

    def train(self, dataset, epochs, probe_set, probe_train):
        # Computation graph definition
        batched_dataset = dataset.batch(self.batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)
        optimizer = tf.compat.v1.train.MomentumOptimizer(0.001, 0.01, 0.9)

        # Main training loop, needs adjustments depending on how training data is handled
        #print(self.visible_bias)
        for epoch in range(epochs):
            if self.anneal:
                self.lr_weights = self.lr_weights / (1 + epoch / self.anneal_val)
                self.lr_vb = self.lr_vb / (1 + epoch / self.anneal_val)
                self.lr_hb = self.lr_hb / (1 + epoch / self.anneal_val)

            if epoch == 42:
                self.k = 3

            num_pts = 0

            print()
            print("Epoch " + str(epoch + 1) + "/" + str(epochs))
            prog_bar = tf.keras.utils.Progbar(32000, stateful_metrics=["train_rmse"])

            try:
                while True:

                    x, r1, r2, r3 = self.get_rx(iterator)
                    x_hot = tf.one_hot(x, self.num_rat, axis=1)
                    grads = self.learn(x_hot, r1, r2, r3)
                    optimizer.apply_gradients(zip(grads, self.get_variables()))

                    # self.apply_gradients(grads)
                    num_pts += 1
                    train_rmse = tf.sqrt(tf.scalar_mul(1 / tf.compat.v1.to_float(tf.math.count_nonzero(tf.add(x, 1))),
                                                       tf.reduce_sum(tf.square(tf.subtract(self.forward(x_hot, r1, r2, r3),
                                                                                 tf.compat.v1.to_float(tf.add(x, 1)))))))
                    prog_bar.update(self.batch_size * num_pts, [("train_rmse", train_rmse)])
            except tf.errors.OutOfRangeError:
                ds = dataset.shuffle(32000)
                batched_dataset = ds.batch(self.batch_size)
                iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)

            predictions, RMSE, MAE = self.pred_with_RMSE(probe_set, probe_train)
            prog_bar.update(self.batch_size * num_pts, [("test_rmse", RMSE), ("test_mae", MAE)])

        save('./tmp/weights.npy', self.weights.numpy())
        save('./tmp/hidden_bias.npy', self.hidden_bias.numpy())
        save('./tmp/visible_bias.npy', self.visible_bias.numpy())
        save('./tmp/D_1.npy', self.D_1.numpy())
        save('./tmp/D_2.npy', self.D_2.numpy())
        save('./tmp/D_3.npy', self.D_3.numpy())

    def pred_for_sub(self, test_set, pred_set, submit=True, filename="rbm.txt"):
        test_set = test_set.repeat(1)
        test_set = test_set.batch(self.batch_size)
        test_iterator = test_set.make_one_shot_iterator()
        pred_set = pred_set.batch(self.batch_size)
        pred_iterator = pred_set.make_one_shot_iterator()

        x, r1, r2, r3 = self.get_rx(pred_iterator)

        x_hot = tf.one_hot(x, self.num_rat, axis=1)
        batch_count = 0
        curr_preds = self.forward(x_hot, r1, r2, r3, False)


        batch_count = 0
        total_predictions = []
        actual = []
        try:
            while True:
                row_batch = test_iterator.get_next()
                test_preds = tf.gather_nd(curr_preds, row_batch.indices)

                total_predictions = tf.concat([total_predictions, test_preds], 0)
                x, r1, r2, r3 = self.get_rx(pred_iterator)

                x_hot = tf.one_hot(x, self.num_rat, axis=1)
                curr_preds = self.forward(x_hot, r1, r2, r3, False)

        except tf.errors.OutOfRangeError:
            pass

        if submit:
            submission = total_predictions.numpy()
            np.savetxt(filename, submission, delimiter="\n")

        return total_predictions

    def pred_with_RMSE(self, test_set, pred_set):
        test_set = test_set.repeat(1280)
        test_set = test_set.batch(1280)
        test_iterator = tf.compat.v1.data.make_one_shot_iterator(test_set)
        pred_set = pred_set.batch(1280)
        pred_iterator = tf.compat.v1.data.make_one_shot_iterator(pred_set)

        x, r1, r2, r3 = self.get_rx(pred_iterator)
        x_hot = tf.one_hot(x, self.num_rat, axis=1)
        curr_preds = self.forward(x_hot, r1, r2, r3, False)


        total_predictions = []
        actual = []
        try:
            while True:
                row_batch = test_iterator.get_next()
                test_preds = tf.gather_nd(curr_preds, row_batch.indices)
                print(test_preds)

                total_predictions = tf.concat([total_predictions, test_preds], 0)
                actual = tf.concat([actual, row_batch.values], 0)
                x, r1, r2, r3 = self.get_rx(pred_iterator)
                x_hot = tf.one_hot(x, self.num_rat, axis=1)
                curr_preds = self.forward(x_hot, r1, r2, r3, False)

        except tf.errors.OutOfRangeError:
            pass

        # pred_rmse = tf.math.sqrt(tf.losses.mean_squared_error(total_predictions, actual))
        pred_rmse = tf.math.sqrt(tf.compat.v1.losses.mean_squared_error(total_predictions, actual))
        pred_mae = tf.keras.losses.MAE(total_predictions, actual)

        return total_predictions, pred_rmse, pred_mae

    def forward(self, visibles, r1, r2, r3, should_mask=True):
        hidden_samples = self.sample_h_given_v(visibles, r1, r2, r3, False)
        visible_samples = self.sample_v_given_h(hidden_samples)

        reduced_visibles = tf.reduce_sum(visibles, axis=1)

        scale = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]])

        if not should_mask:
            return tf.tensordot(scale, visible_samples, [[1], [1]])[0]

        mask = tf.where(tf.equal(reduced_visibles, 0), tf.zeros_like(reduced_visibles), tf.ones_like(reduced_visibles))
        mask = tf.stack([mask, mask, mask, mask, mask], axis=1)
        return tf.tensordot(scale, mask * visible_samples, [[1], [1]])[0]

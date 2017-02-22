import tensorflow as tf
import numpy as np

from datasets.twitter import data
from datasets.twitter import data_utils

class enc_lm_seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
            model_name= 'enc_langmodel',
            ckpt_path= 'ckpt/enc_langmodel/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        def __graph__():
            # you know what this means
            tf.reset_default_graph()
            #
            # placeholders
            xs_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='xs')
            ys_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='ys') # decoder targets
            dec_inputs_ = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                        name='dec_inputs')

            # embed encoder input
            embs = tf.get_variable('emb', [vocab_size, state_size])
            enc_inputs = tf.nn.embedding_lookup(embs, xs_)

            # embed decoder input
            dec_inputs = tf.nn.embedding_lookup(embs, dec_inputs_)

            # define basic lstm cell
            basic_cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
            # add dropout
            #   dropout's keep probability
            keep_prob_ = tf.placeholder(tf.float32)
            basic_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=keep_prob_)

            # stack cells
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)

            with tf.variable_scope('encoder') as scope:
                # define encoder
                enc_op, enc_context = tf.nn.dynamic_rnn(cell=stacked_lstm, dtype=tf.float32, 
                                                  inputs=enc_inputs)

            ###
            # project enc_op
            Ve = tf.get_variable('Ve', shape=[state_size, vocab_size], 
                                initializer=tf.contrib.layers.xavier_initializer())
            be = tf.get_variable('be', shape=[vocab_size], 
                                 initializer=tf.constant_initializer(0.))
            ###
            # reshape enc_op
            enc_op_reshaped = tf.reshape(enc_op, [-1, state_size])
            enc_logits = tf.matmul(enc_op_reshaped, Ve) + be

            # optimization
            enc_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=enc_logits,
                                                                    labels=tf.reshape(xs_, [-1]))
            enc_loss = tf.reduce_mean(enc_losses)
            enc_train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(enc_loss)


            with tf.variable_scope('decoder') as scope:
                # define decoder 
                dec_op, _ = tf.nn.dynamic_rnn(cell=stacked_lstm, dtype=tf.float32,
                                              initial_state= enc_context,
                                              inputs=dec_inputs)
                
            ###    
            # predictions
            Vd = tf.get_variable('Vd', shape=[state_size, vocab_size], 
                                initializer=tf.contrib.layers.xavier_initializer())
            bd = tf.get_variable('bd', shape=[vocab_size], 
                                 initializer=tf.constant_initializer(0.))
            ####
            # flatten states to 2d matrix for matmult with V
            dec_op_reshaped = tf.reshape(dec_op, [-1, state_size])
            # /\_o^o_/\
            dec_logits = tf.matmul(dec_op_reshaped, Vd) + bd
            #
            # predictions
            predictions = tf.nn.softmax(dec_logits)
            #
            # optimization
            dec_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_logits,
                                                                    labels=tf.reshape(ys_, [-1]))
            dec_loss = tf.reduce_mean(dec_losses)
            dec_train_op = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(dec_loss)
            #
            # joint training
            loss = enc_loss + dec_loss
            train_op = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
            #
            # attach symbols to class
            self.loss = loss
            self.enc_loss = enc_loss
            self.dec_loss = dec_loss
            self.train_op = train_op
            self.enc_train_op = enc_train_op
            self.dec_train_op = dec_train_op
            self.predictions = predictions
            self.keep_prob_ = keep_prob_
            self.xs_ = xs_
            self.ys_ = ys_
            self.dec_inputs_ = dec_inputs_
            #####
        ####
        # build graph
        __graph__()

    def train_joint(self, trainset, testset, epochs=100, n=100):

        def fetch_dict(datagen, keep_prob=0.5):
            bx, by = datagen.__next__()
            by_dec = np.zeros_like(by).T
            by_dec[1:] = by.T[:-1]
            by_dec = by_dec.T
            feed_dict = { 
                    self.xs_ : bx, 
                    self.ys_ : by,
                    self.dec_inputs_ : by_dec,
                    self.keep_prob_ : keep_prob
                    }
            return feed_dict

        ##
        # setup session
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        try:
            # start training
            for j in range(epochs):
                mean_loss = 0
                for i in range(n):
                    _, l = sess.run([self.train_op, self.loss], 
                            feed_dict = fetch_dict(trainset) 
                            )
                    mean_loss += l

                print(f'>> [{j}] train loss at : {mean_loss/n}')
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                #
                # evaluate
                testloss = sess.run([self.dec_loss], 
                        feed_dict = fetch_dict(testset, keep_prob=1.)
                        )
                print(f'test loss : {testloss}')
 
        except KeyboardInterrupt:
            print(f'\n>> Interrupted by user at iteration {j}')


    def train_alternate(self, trainset, testset, epochs=100, n=100):

        def fetch_dict(datagen, keep_prob=0.5):
            bx, by = datagen.__next__()
            by_dec = np.zeros_like(by).T
            by_dec[1:] = by.T[:-1]
            by_dec = by_dec.T
            feed_dict = { 
                    self.xs_ : bx, 
                    self.ys_ : by,
                    self.dec_inputs_ : by_dec,
                    self.keep_prob_ : keep_prob
                    }
            return feed_dict

        ##
        # setup session
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # get last checkpoint
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # verify it
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        try:
            # start training
            for j in range(epochs):
                mean_loss = 0
                for i in range(n): # train decoder loss with 70% probability
                    if np.random.rand() < 0.7: 
                        _, l = sess.run([self.dec_train_op, self.dec_loss],
                                feed_dict = fetch_dict(trainset) 
                                )
                        mean_loss += l
                    else: # train encoder lang model with 30% probability
                        _, l = sess.run([self.enc_train_op, self.enc_loss],
                                feed_dict = fetch_dict(trainset) 
                                )
                        mean_loss += l

                print(f'>> [{j}] train loss at : {mean_loss/n}')
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                #
                # evaluate
                testloss = sess.run([self.dec_loss], 
                        feed_dict = fetch_dict(testset, keep_prob=1.)
                        )
                print(f'test loss : {testloss}')
 
        except KeyboardInterrupt:
            print(f'\n>> Interrupted by user at iteration {j}')


if __name__ == '__main__':
    #
    # gather data
    metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
    # split data
    (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
    #
    # prepare train set generator
    #  set batch_size
    batch_size = 16
    trainset = data_utils.rand_batch_gen(trainX, trainY, batch_size)
    testset = data_utils.rand_batch_gen(testX, testY, batch_size=1024)

    ###
    # infer vocab size
    vocab_size = len(metadata['idx2w'])
    #
    # create a model
    model = enc_lm_seq2seq(state_size=1024, vocab_size=vocab_size, num_layers=3)
    # train
    model.train_alternate(trainset, testset, n=1000)

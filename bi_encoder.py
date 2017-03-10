import tensorflow as tf
import numpy as np

from datasets.twitter import data
from datasets.twitter import data_utils

import sys

class bienc_seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
            model_name= 'bienc_seq2seq',
            ckpt_path= 'ckpt/bienc_seq2seq/'):

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        def __graph__():
            # you know what this means
            tf.reset_default_graph()


            # placeholders
            xs_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='xs')
            ys_ = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                 name='ys') # decoder targets
            dec_inputs_length_ = tf.placeholder(dtype=tf.int32, shape=[None,],
                                        name='dec_inputs_length')


            # embed encoder input
            embs = tf.get_variable('emb', [vocab_size, state_size])
            enc_inputs = tf.nn.embedding_lookup(embs, xs_)

            
            # define lstm cell for encoder
            encoder_cell = tf.contrib.rnn.LSTMCell(state_size)
            # add dropout
            #   dropout's keep probability
            keep_prob_ = tf.placeholder(tf.float32)
            encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, 
                    output_keep_prob=keep_prob_)
            # stack cells
            encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell]*num_layers,
                    state_is_tuple=True)

            init_state = encoder_cell.zero_state(batch_size=tf.shape(xs_)[0], dtype=tf.float32)

            # transpose encoder inputs to time-major
            enc_inputs = tf.transpose(enc_inputs, [1,0,2])
            #
            # der bi encoder
            with tf.variable_scope('encoder-fw') as scope: # forward sequence
                enc_states_fw = tf.scan(lambda st_1, x : encoder_cell(x, st_1)[-1], 
                        enc_inputs, initializer=init_state)

            with tf.variable_scope('encoder-bw') as scope: # backward sequence
                enc_states_bw = tf.scan(lambda st_1, x : encoder_cell(x, st_1)[-1], 
                                    tf.reverse(enc_inputs, axis=[0]), # <- reverse inputs
                                    initializer=init_state)

            # project context
            Wc = tf.get_variable('Wc', shape=[2, num_layers, state_size*2, state_size], 
                                initializer=tf.contrib.layers.xavier_initializer())

            # extract context [get final state; project c,h to [state_size]; list->tuple]
            enc_final_state = []
            for layer in range(num_layers):
                enc_c = tf.concat( (enc_states_fw[layer].c[-1], enc_states_bw[layer].c[-1]), 1)
                enc_c = tf.matmul(enc_c, Wc[0][layer])
                enc_h = tf.concat( (enc_states_fw[layer].h[-1], enc_states_bw[layer].h[-1]), 1)
                enc_h = tf.matmul(enc_h, Wc[1][layer])
                enc_final_state.append(tf.contrib.rnn.LSTMStateTuple(c = enc_c,h = enc_h))
            # convert list to tuple - eww!
            enc_final_state = tuple(enc_final_state)

            '''
            the realm of der decoder begins here
            '''
            ###    
            # output projection
            V = tf.get_variable('V', shape=[state_size, vocab_size], 
                                initializer=tf.contrib.layers.xavier_initializer())
            bo = tf.get_variable('bo', shape=[vocab_size], 
                                 initializer=tf.constant_initializer(0.))

            ###
            # embedding for pad symbol
            PAD = tf.nn.embedding_lookup(embs, tf.zeros(shape=[tf.shape(xs_)[0],], 
                        dtype=tf.int32))

            ###
            # init function for raw_rnn
            def loop_fn_initial(time, cell_output, cell_state, loop_state):
                assert cell_output is None and loop_state is None and cell_state is None

                elements_finished = (time >= dec_inputs_length_)
                initial_input = PAD
                initial_cell_state = enc_final_state
                initial_loop_state = None
                
                return (elements_finished,
                        initial_input,
                        initial_cell_state,
                        None,
                        initial_loop_state)

            ###
            # state transition function for raw_rnn
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_state is None:
                    return loop_fn_initial(time, cell_output, cell_state, loop_state)
                
                emit_output = cell_output
                next_cell_state = cell_state

                elements_finished = (time >= dec_inputs_length_)
                finished = tf.reduce_all(elements_finished)

                def search_for_next_input():
                    output = tf.matmul(cell_output, V) + bo
                    prediction = tf.argmax(output, axis=1)
                    return tf.nn.embedding_lookup(embs, prediction)

                
                next_input = tf.cond(finished, lambda : PAD, search_for_next_input)

                next_loop_state = None

                return (elements_finished, 
                        next_input, 
                        next_cell_state,
                        emit_output,
                        next_loop_state)
                

            ###
            # define the decoder with raw_rnn <- loop_fn, loop_fn_initial
            decoder_cell = tf.contrib.rnn.LSTMCell(state_size)
            decoder_cell = tf.contrib.rnn.MultiRNNCell([decoder_cell]*num_layers, 
                    state_is_tuple=True)

            with tf.variable_scope('decoder') as scope:
                decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()

            ####
            # flatten states to 2d matrix for matmult with V
            dec_op_reshaped = tf.reshape(decoder_outputs, [-1, state_size])
            # /\_o^o_/\
            logits = tf.matmul(dec_op_reshaped, V) + bo
            #
            # predictions
            predictions = tf.nn.softmax(logits)
            #
            # optimization
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=tf.reshape(ys_, [-1]))
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
            #
            # attach symbols to class
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.keep_prob_ = keep_prob_ # placeholders
            self.xs_ = xs_
            self.ys_ = ys_
            self.dec_inputs_length_ = dec_inputs_length_
            #####
        ####
        # build graph
        __graph__()

    def train(self, trainset, testset, epochs=100, n=100):

        def fetch_dict(datagen, keep_prob=0.5):
            bx, by = datagen.__next__()
            bx, by = bx.T, by.T
            enc_lengths = (bx != 0).sum(1)
            dec_lengths = (by != 0).sum(1)
            by = by.T[:dec_lengths.max()]
            bx = bx.T[:enc_lengths.max()].T
            feed_dict = { 
                    self.xs_ : bx, 
                    self.ys_ : by,
                    self.dec_inputs_length_ : dec_lengths,
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
                    sys.stdout.write('[{}/{}]\r'.format(i,n))
                    sys.stdout.flush()

                print('>> [{}] train loss at : {}'.format(j, mean_loss/n))
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                #
                # evaluate
                testloss = sess.run([self.loss], 
                        feed_dict = fetch_dict(testset, keep_prob=1.)
                        )
                print('test loss : {}'.format(testloss))
 
        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))


if __name__ == '__main__':
    #
    # gather data
    metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
    # split data
    (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
    #
    # prepare train set generator
    #  set batch_size
    batch_size = 128
    trainset = data_utils.rand_batch_gen(trainX, trainY, batch_size)
    testset = data_utils.rand_batch_gen(testX, testY, batch_size=1024)

    ###
    # infer vocab size
    vocab_size = len(metadata['idx2w'])
    #
    # create a model
    model = bienc_seq2seq(state_size=1024, vocab_size=vocab_size, num_layers=3)
    # train
    model.train(trainset, testset, n=100, epochs=100000)

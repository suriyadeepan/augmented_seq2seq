import tensorflow as tf
import numpy as np

from datasets.friends import data
from datasets.friends import data_utils

import sys

class contextual_seq2seq(object):

    def __init__(self, state_size, vocab_size, num_layers,
            ext_context_size, 
            model_name= 'contextual_seq2seq',
            ckpt_path= 'ckpt/contextual_seq2seq/'):

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
            ext_context_ = tf.placeholder(dtype=tf.int32, shape=[None,],
                                        name='ext_context') 


            # embed encoder input
            embs = tf.get_variable('emb', [vocab_size, state_size])
            enc_inputs = tf.nn.embedding_lookup(embs, xs_)

            
            # embed ext_context
            ext_ctxt_embs = tf.get_variable('ext_ctxt_emb', [ext_context_size, state_size])
            ext_context = tf.nn.embedding_lookup(ext_ctxt_embs, ext_context_)


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

            with tf.variable_scope('encoder') as scope:
                # define encoder
                enc_op, enc_context = tf.nn.dynamic_rnn(cell=encoder_cell, dtype=tf.float32, 
                                                  inputs=enc_inputs)

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
                initial_cell_state = enc_context
                initial_loop_state = None
                
                return (elements_finished,
                        initial_input,
                        initial_cell_state,
                        None,
                        initial_loop_state)

            ###
            # state transition function for raw_rnn
            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_state is None:    # time == 0
                    return loop_fn_initial(time, cell_output, cell_state, loop_state)
                
                emit_output = cell_output  # == None for time == 0
                #
                # couple external context with cell states (c, h)
                next_cell_state = []
                for layer in range(num_layers):
                    next_cell_state.append(tf.contrib.rnn.LSTMStateTuple( 
                            c = cell_state[layer].c + ext_context,
                            h = cell_state[layer].h + ext_context
                            ))

                next_cell_state = tuple(next_cell_state)

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
            self.ext_context_size = ext_context_size
            self.keep_prob_ = keep_prob_ # placeholders
            self.xs_ = xs_
            self.ys_ = ys_
            self.dec_inputs_length_ = dec_inputs_length_
            self.ext_context_ = ext_context_
            #####
        ####
        # build graph
        __graph__()

    def train(self, trainset, testset, n, epochs):

        print('\n>> Training begins!\n')

        def fetch_dict(datagen, keep_prob=0.5):
            bx, by, br = datagen.__next__()
            while ( not (bx.shape[0] > 0 and bx.shape[1] > 0 and by.shape[0] > 0 and by.shape[1] > 0 and br.shape[0] > 0) ):
                bx, by, br = datagen.__next__()

            dec_lengths = np.full([bx.shape[0],], by.shape[1], dtype=np.int32)

            feed_dict = { 
                    self.xs_ : bx, 
                    self.ys_ : by,
                    self.dec_inputs_length_ : dec_lengths,
                    self.ext_context_ : br,
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

                print('\n>> [{}] train loss at : {}'.format(j, mean_loss/n))
                saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                #
                # evaluate
                testloss = 0
                for k in range(300):
                    testloss += sess.run([self.loss], 
                            feed_dict = fetch_dict(testset, keep_prob=1.)
                            )[0]
                print('test loss : {}'.format(testloss/300)) # make this changeable
 
        except KeyboardInterrupt:
            print('\n>> Interrupted by user at iteration {}'.format(j))


if __name__ == '__main__':
    #
    # gather data
    data_, metadata = data.load_data(PATH='datasets/friends/')
    # split into batches
    batches = data_utils.create_batches(data_)
    # split data
    traindata, testdata = data_utils.split_dataset(batches)
    # prepare train set generator
    #  
    trainset = data_utils.rand_batch_gen(traindata)
    testset = data_utils.rand_batch_gen(testdata)

    ###
    # infer vocab size
    vocab_size = len(metadata['idx2w'])
    ext_context_size = metadata['respect_size']
    #
    # create a model
    model = contextual_seq2seq(state_size=1024, vocab_size=vocab_size, 
            num_layers=3, ext_context_size=ext_context_size)
    # train
    model.train(trainset, testset, n=1000, epochs=1000000)

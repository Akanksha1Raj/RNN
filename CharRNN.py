from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    '''
    choose top_n most possible charactors in predictions
    this can help reduce some noise
    inputs:
    preds
    vocab_size
    top_n
    '''
    p = np.squeeze(preds)
    # set all values other that top_n choices to 0
    p[np.argsort(p)[:-top_n]] = 0
    # normalization
    p = p / np.sum(p)
    # randomly choose one
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class CharRNN():
    def __init__(self, num_classes, batch_size=64, num_steps=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, 
                 grad_clip=5, train_keep_prob=0.5, sampling=False):
        '''
        Initialize the input parameter to define the network
        inputs:
        :param num_classes: (int) the vocabulary size of your input data
        :param batch_size: (int) number of sequences in one batch
        :param num_steps: (int) length of each seuqence in one batch
        :param cell_type: your rnn cell type, 'LSTM' or 'GRU'
        :param rnn_size: (int) number of units in one rnn layer
        :param num_layers: (int) number of rnn layers
        :param learning_rate: (float)
        :param grad_clip: constraint of gradient to avoid gradient explosion
        :param train_keep_prob: (float) dropout probability for rnn cell training
        :param sampling: (boolean) whether train mode or sample mode
        '''
        #print("##############")
        # if not training
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            #print("yet to go")
            batch_size, num_steps = batch_size, num_steps

        #print("not yet")
        tf.reset_default_graph()
       
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.sampling=sampling
        
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.my_optimizer()
        self.saver = tf.train.Saver()
        
        #print("#######____")
        
    
    
    def inputs_layer(self):
        '''
        build the input layer
        '''
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        
        # add keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # one_hot encoding
        self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
    
    
    def rnn_layer(self):
        '''
        build rnn_cell layer
        we will use the paramters:
        self.cell_type, self.rnn_size, self.keep_prob, self.num_layers,
        self.batch_size, self.rnn_inputs
        we have to define:
        self.rnn_outputs, self.final_state for later use
        '''
        #########################################################################################################
        #           TODO: finish the rnn layer definition, you should enable the selection of cell type         #
        cell=[]
        #print(type(self.batch_size))
        if (self.cell_type=='GRU'):
            for i in range(self.num_layers):
                gru = tf.nn.rnn_cell.GRUCell(self.rnn_size)
                #print("gru")
                if self.sampling:
                    if self.train_keep_prob < 1:
                          cell.append(tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=self.train_keep_prob))
                else:
                      cell.append(gru)
       
        elif (self.cell_type=='LSTM'):
            
            for i in range(self.num_layers):
                lstm = tf.nn.rnn_cell.LSTMCell(self.rnn_size) 
                #print("lstm")
                if self.sampling:
                    if self.train_keep_prob < 1:
                          cell.append(tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.train_keep_prob))
                else:
                    cell.append(lstm)
        else:
            raise Exception("type of the cell is undefined")
        #print("hello")
        #print(cell.shape)
        #print(type[cell[0]])
        cell = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.rnn_inputs,initial_state=self.initial_state)
            
        #########################################################################################################
        ##raise NotImplementedError('Please edit this function.')
    
    
    def outputs_layer(self):
        ''' 
        build the output layer
        '''
        # concate the output of rnn_cell，example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        seq_output = tf.concat(self.rnn_outputs, axis=1) # tf.concat(concat_dim, values)
        # reshape
        x = tf.reshape(seq_output, [-1, self.rnn_size])
        
        # define softmax layer variables:
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
        
        # calculate logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        
        # softmax generate probability predictions
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')
        
        
    def my_loss(self):
        '''
        calculat loss according to logits and targets
        '''
        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        
        
    def my_optimizer(self):
        '''
        build our optimizer
        Unlike previous worries of gradient vanishing problem,
        for some structures of rnn cells, the calculation of hidden layers' weights 
        may lead to an "exploding gradient" effect where the value keeps growing.
        To mitigate this, we use the gradient clipping trick. Whenever the gradients are updated, 
        they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
        parameters we will use:
        self.loss, self.grad_clip, self.learning_rate
        we have to define:
        self.optimizer for later use
        '''
        # using clipping gradients
        #######################################################
        # TODO: implement your optimizer with gradient clipping
        gradient, _ = tf.clip_by_global_norm(tf.gradients(self.loss,tf.trainable_variables()),  self.grad_clip)
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradient, tf.trainable_variables()))
    
        #######################################################
        #raise NotImplementedError('Please edit this function.')
        
        
    def train(self, batches, max_count, save_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)
            # Train network
            for x, y in batches:
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
               
                batch_loss, new_state, _ = sess.run([self.loss, 
                                                     self.final_state, 
                                                     self.optimizer
                                                    ], 
                                                     feed_dict=feed)
                    
                end = time.time()
                if counter % 200 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    
                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
                    
                if counter >= max_count:
                    break
            
            self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
               
        
    def sample(self, checkpoint, n_samples, vocab_size, vocab_to_ind, ind_to_vocab, prime='You \n'):
        '''
        generate new text given the prime word
        inputs:
        :param n_samples: (int) number of characters you want to generate
        :param vocab_size: (int) number of vocabulary size of your input data
        :param vocab_to_ind, ind_to_vocab: mapping from unique characters to indices
        :param prime: (str) you new text starting word
        outputs:
        -a string of generated characters
        '''
       
        # change text into character list
        samples = [c for c in prime]
        #####################################################################################
        # TODO: implement sampling function
        # Hint: you should restore from saved checkpoint, start the model in sampling mode on 
        # prime word first and then generate new characters, remember to use pick_top_n to
        # reduce the noise.
        #in=[]
        #for char in prime:
        #   in.append(char)
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
         
        s=self.session.run(self.initial_state)
       
        str1 = prime + " "
        input = np.zeros((1, 1))
        #print("test")
        input[0, 0] = vocab_to_ind[samples[-1]]
        feed={self.inputs: input,
                self.keep_prob: 1.,
                self.initial_state: s}
        p = np.zeros((vocab_size, ))
        p, s = self.session.run([self.prob_pred, self.final_state],
                                        feed_dict=feed)
        ch = pick_top_n(p, vocab_size,3)
        
        str1 +=ind_to_vocab[ch]
        #print("test2")
        for i in range(n_samples):
           
            input[0, 0] = ch
            feed={self.inputs: input,
                    self.keep_prob: 1.,
                    self.initial_state: s}
            preds, s = self.session.run([self.prob_pred, self.final_state],
                                        feed_dict=feed)
            ch = pick_top_n(p, vocab_size,3)
            str1 +=ind_to_vocab[ch]
            
        return str1 
        #####################################################################################
        raise NotImplementedError('Please edit this function.')
    
    
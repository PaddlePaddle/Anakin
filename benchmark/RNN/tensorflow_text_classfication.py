
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import timeit

# In[2]:

def language_run(data_set):
    voc_size=566227
    hidden_size=128
    hidden_size_after_lstm=96
    hidden_size_after_fc=2
    batch_size=1
    tf.device('/cpu:0')


    # In[3]:


    x_input = tf.placeholder(
        tf.int32, [1,None], name="x_input")


    # In[4]:


    embedding_table = tf.get_variable('emb', [voc_size, hidden_size], dtype=tf.float32)
    embedding_out=tf.nn.embedding_lookup(embedding_table, x_input)


    # In[5]:


    lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size)
    # lstm_init_state=lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # lstm_out,_=tf.nn.dynamic_rnn(lstm_cell,embedding_out,initial_state=lstm_init_state)
    (output_fw, output_bw), _=tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                                                              lstm_cell, embedding_out,
                                                              dtype=tf.float32)

    bi_lstm_out = tf.concat([output_fw, output_bw], axis=-1)

    # In[6]:


    fc_weights = tf.get_variable(
        'fc_weights', [ hidden_size*2,hidden_size_after_lstm],
        initializer=tf.truncated_normal_initializer(
            stddev=0.01, dtype=tf.float32),
        dtype=tf.float32)
    fc_bias = tf.get_variable(
        'fc_bias', [hidden_size_after_lstm],
        initializer=tf.truncated_normal_initializer(
            stddev=0.0, dtype=tf.float32),
        dtype=tf.float32)
    bi_lstm_out=tf.squeeze(bi_lstm_out,[0])
    fc1_out=tf.tanh(tf.matmul(bi_lstm_out,fc_weights) + fc_bias)

    # In[7]:
    fc2_weights = tf.get_variable(
        'fc2_weights', [ hidden_size_after_lstm,hidden_size_after_fc],
        initializer=tf.truncated_normal_initializer(
            stddev=0.01, dtype=tf.float32),
        dtype=tf.float32)
    fc2_bias = tf.get_variable(
        'fc2_bias', [hidden_size_after_fc],
        initializer=tf.truncated_normal_initializer(
            stddev=0.0, dtype=tf.float32),
        dtype=tf.float32)
    fc2_out=tf.matmul(fc1_out,fc2_weights) + fc2_bias

    softmax=tf.nn.softmax(fc2_out)


    # In[8]:

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # In[9]:


    def clock(func):
        def clocked(*args):
            t0 = timeit.default_timer()
            result = func(*args)
            elapsed = timeit.default_timer() - t0
            name = func.__name__
            arg_str = ', '.join(repr(arg) for arg in args)
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, 'arg_str', result))
            lines=len(args[0])
            counter=sum(len(line) for line in args[0])
            print('Delay = '+str(elapsed*1000/lines)+'ms')
            return result
        return clocked


    # In[10]:


    @clock
    def benchmark(data_set):
        for one_batch in data_set:
            sess.run([softmax],{x_input:np.array(one_batch).reshape(1,len(one_batch))})

            # tf.train.write_graph(sess.graph.as_graph_def(), 'model/text_classfi_model_tf/', 'graph.pb', as_text=False)
            # saver=tf.train.Saver()
            # saver.save(sess, "model/text_classfi_model_tf/model.cpkt")
            # exit()


    benchmark(data_set)
if __name__=='__main__':
    import getopt
    import sys
    proc_num=1
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "process_num="])
        for key,arg in opts:
            if key in ('-h','--help'):
                print('usage --process_num=k ,default=1')
            if key in ('--process_num'):
                proc_num=int(arg)
        print(opts)
    except getopt.GetoptError:
        pass

    from read_ptb_data import PTB_Data_Reader
    data_set=PTB_Data_Reader().read()
    word_sum=sum(len(i) for i in data_set)
    from multiprocessing import Process
    threads=[]
    t0 = timeit.default_timer()
    for i in range(proc_num):
        t =Process(target=language_run,args=(data_set,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed = timeit.default_timer() - t0
    print(__file__,'process = ',proc_num,',QPS = ',len(data_set)/elapsed*proc_num,' line / second ,',word_sum/elapsed*proc_num,'words/second')
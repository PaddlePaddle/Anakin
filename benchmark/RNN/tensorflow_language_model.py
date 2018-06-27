
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import timeit

# In[2]:

def language_run(data_set):
    voc_size=10001
    hidden_size=200
    batch_size=1
    tf.device('/cpu:0')


    # In[3]:


    x_input = tf.placeholder(
        tf.int32, [1,None], name="x_input")
    # x_input_len = tf.placeholder(
    #                 tf.int32, name="x_input_len")


    # In[4]:


    embedding_table = tf.get_variable('emb', [voc_size, hidden_size], dtype=tf.float32)
    embedding_out=tf.nn.embedding_lookup(embedding_table, x_input)


    # In[5]:


    gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
    gru_init_state=gru_cell.zero_state(batch_size, dtype=tf.float32)
    gru_out,_=tf.nn.dynamic_rnn(gru_cell,embedding_out,initial_state=gru_init_state)


    # In[6]:


    fc_weights = tf.get_variable(
        'fc_weights', [ hidden_size,voc_size],
        initializer=tf.truncated_normal_initializer(
            stddev=0.01, dtype=tf.float32),
        dtype=tf.float32)
    fc_bias = tf.get_variable(
        'fc_bias', [voc_size],
        initializer=tf.truncated_normal_initializer(
            stddev=0.0, dtype=tf.float32),
        dtype=tf.float32)
    gru_out=tf.squeeze(gru_out,[0])
    fc_out=tf.matmul(gru_out,fc_weights) + fc_bias


    # In[7]:


    softmax=tf.nn.softmax(fc_out)


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

            # tf.train.write_graph(sess.graph.as_graph_def(), 'model/language_model_tf/', 'graph.pb', as_text=False)
            # saver=tf.train.Saver()
            # saver.save(sess, "model/language_model_tf/model.cpkt")
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
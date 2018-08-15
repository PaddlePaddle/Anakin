
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time
import timeit

# In[2]:

def language_run(data_set):
    word_voc_size=1942562
    mention_voc_size=57
    word_hidden_size=32
    mention_hidden_size=20
    gru_hidden_size=36

    fc1_hidden_size=49


    batch_size=1
    tf.device('/cpu:0')


    # In[3]:


    x_input = tf.placeholder(
        tf.int32, [1,None], name="x_input")
    x_input_len = tf.placeholder(
        tf.int32, [None],name="x_input_len")
    mention_input = tf.placeholder(
        tf.int32, [1,None], name="mention_input")

    # In[4]:


    embedding_table_word_r = tf.get_variable('emb_w_r', [word_voc_size, word_hidden_size], dtype=tf.float32)
    embedding_out_r=tf.nn.embedding_lookup(embedding_table_word_r, x_input)

    embedding_table_mention_r = tf.get_variable('emb_m_r', [mention_voc_size, mention_hidden_size], dtype=tf.float32)
    embedding_mention_out_r=tf.nn.embedding_lookup(embedding_table_mention_r, mention_input)
    ##
    embedding_table_word_l = tf.get_variable('emb_w_l', [word_voc_size, word_hidden_size], dtype=tf.float32)
    embedding_out_l=tf.nn.embedding_lookup(embedding_table_word_l, x_input)

    embedding_table_mention_l = tf.get_variable('emb_m_l', [mention_voc_size, mention_hidden_size], dtype=tf.float32)
    embedding_mention_out_l=tf.nn.embedding_lookup(embedding_table_mention_l, mention_input)

    emb_r=tf.concat([embedding_out_r,embedding_mention_out_r],axis=-1)
    emb_l=tf.concat([embedding_out_l,embedding_mention_out_l],axis=-1)
    # In[5]:
    with tf.variable_scope('forward'):
        gru_cell_r = tf.contrib.rnn.GRUCell(gru_hidden_size)
        gru_init_state_r = gru_cell_r.zero_state(batch_size, dtype=tf.float32)
        gru_out_r, _ = tf.nn.dynamic_rnn(gru_cell_r, emb_r, initial_state=gru_init_state_r)

    with tf.variable_scope('backward'):
        gru_cell_l = tf.contrib.rnn.GRUCell(gru_hidden_size)
        gru_init_state_l = gru_cell_l.zero_state(batch_size, dtype=tf.float32)
        gru_out_l, _ = tf.nn.dynamic_rnn(gru_cell_l, emb_l, initial_state=gru_init_state_l)

    bi_gru_out=tf.concat([gru_out_l,gru_out_r],axis=-1)

    # In[6]:


    fc_weights = tf.get_variable(
        'fc_weights', [ gru_hidden_size*2,fc1_hidden_size],
        initializer=tf.truncated_normal_initializer(
            stddev=0.01, dtype=tf.float32),
        dtype=tf.float32)
    fc_bias = tf.get_variable(
        'fc_bias', [fc1_hidden_size],
        initializer=tf.truncated_normal_initializer(
            stddev=0.0, dtype=tf.float32),
        dtype=tf.float32)
    bi_gru_out=tf.squeeze(bi_gru_out,[0])
    fc1_out=tf.matmul(bi_gru_out,fc_weights) + fc_bias


    # In[7]:
    crf_weights = tf.get_variable(
        'crf_weights', [ fc1_hidden_size,fc1_hidden_size],
        initializer=tf.truncated_normal_initializer(
            stddev=0.01, dtype=tf.float32),
        dtype=tf.float32)

    fc1_out=tf.reshape(fc1_out,[batch_size,-1,fc1_hidden_size])
    crf_out,_=tf.contrib.crf.crf_decode(fc1_out,crf_weights,x_input_len)





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
            word_vec,mention_vec=one_batch[0],one_batch[1]
            sess.run([crf_out],{x_input:np.array(word_vec).reshape(1,len(word_vec)),mention_input:np.array(mention_vec).reshape(1,len(mention_vec)),x_input_len:[len(word_vec)]})

            # tf.train.write_graph(sess.graph.as_graph_def(), 'model/language_model_tf/', 'graph.pb', as_text=False)
            # saver=tf.train.Saver()
            # saver.save(sess, "model/chinese_ner_model_tf/")
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

    from read_ptb_data import NER_Data_Reader
    data_set=NER_Data_Reader().read()
    word_sum=sum(len(i[0]) for i in data_set)
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
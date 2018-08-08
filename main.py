import tensorflow as tf
import numpy as np

tf.reset_default_graph()
sess = tf.InteractiveSession()


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_input = go_step_embedding
    initial_cell_state = encoder_state
    initial_cell_output = None
    initial_loop_state = None
    
    return (initial_elements_finished, initial_input, initial_cell_state,
            initial_cell_output, initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.matmul(previous_output, W) + b
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths)
    finished = tf.reduce_all(elements_finished)
    print(finished)
    ninput = tf.cond(finished, lambda: pad_step_embedding, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None
    
    return (finished, ninput, state, output, loop_state)
    
def l_func(time, poutput, pstate, ploopstate):
    if pstate is None:
        assert pstate is None and poutput is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, poutput, pstate, ploopstate)

if __name__ == "__main__":
    GO = 1
    EOS = 2
    PAD = 0
    
    vocab_size = 10
    input_embedding_size = 20
    
    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units
    
    with tf.name_scope("inputs-targets"):
        encoder_inputs = tf.placeholder(tf.int32, [None, None], name = "Enc_in")
        encoder_input_length = tf.placeholder(tf.int32, [None,], name='Enc_len')
        decoder_targets = tf.placeholder(tf.int32, [None, None], name='Dec_tar')
    
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    
    with tf.variable_scope("encoder") as encoding_scope:
        encoder_cells = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cells, encoder_inputs_embedded,
                                                          dtype=tf.float32,
                                                          time_major = True)
    
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    
    with tf.variable_scope("decoder") as decoder_scope:
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
        decoder_lengths = encoder_input_length + 3
        W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
        b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
        
        assert EOS == 2 and GO == 1 and PAD == 0
        
        go_time_slice = tf.ones([batch_size], dtype=tf.int32, name="GO")
        eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name="EOS") * 2
        pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name="PAD")
        
        go_step_embedding = tf.nn.embedding_lookup(embeddings, go_time_slice)
        eos_step_embedding = tf.nn.embedding_lookup(embeddings, eos_time_slice)
        pad_step_embedding = tf.nn.embedding_lookup(embeddings, pad_time_slice)
        
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn=l_func)
        decoder_outputs = decoder_outputs_ta.stack()
        
        decoder_max_time, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.matmul(decoder_outputs_flat, W) + b
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_time, decoder_batch_size, vocab_size))
        
        decoder_predictions = tf.argmax(decoder_logits, 2)
    
    
    with tf.variable_scope("Training") as train_scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
                                                                logits = decoder_logits)
        
        loss = tf.reduce_mean(cross_entropy)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        
    sess.run(tf.global_variables_initializer())
    tx = np.random.randint(low = 2, high = 9,size = (7, 6))
    ty = np.random.randint(low = 2, high = 9,size = (7, 8))
    #print(sess.run(tf.one_hot(ty, depth=vocab_size)).shape)
    print(sess.run(train_op, feed_dict={encoder_inputs : tx, encoder_input_length : [6] * 6, decoder_targets : ty}))
    sess.close()
    
    
    
    
    
    
    
    
    
    
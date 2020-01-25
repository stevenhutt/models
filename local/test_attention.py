import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Embedding
import numpy as np


dim  = 3
t_q = 5
t_kv = 7
mbs = 2
q = np.random.normal(size=(mbs, t_q, dim)).astype(np.float32)
v = np.random.normal(size=(mbs, t_kv, dim)).astype(np.float32)
k = np.random.normal(size=(mbs, t_kv, dim)).astype(np.float32)

def test_attention():
    q_input = tf.keras.Input(shape=(t_q, dim), dtype='float32', name='q_input')
    v_input = tf.keras.Input(shape=(t_kv, dim), dtype='float32', name='v_input')
    k_input = tf.keras.Input(shape=(t_kv, dim), dtype='float32', name='k_input')

    attention_layer = tf.keras.layers.Attention(name='attention')

    output = attention_layer([q_input, v_input, k_input])

    model = Model(inputs=[q_input, v_input, k_input], outputs=output)

    return model([q, v, k])

def test_attention_2():
    q_input = tf.keras.Input(shape=(t_q, dim), dtype='float32', name='q_input')
    v_input = tf.keras.Input(shape=(t_kv, dim), dtype='float32', name='v_input')
    k_input = tf.keras.Input(shape=(t_kv, dim), dtype='float32', name='k_input')

    # (mbs, t_q, dim) x (mbs, dim, t_kv) -> (mbs, t_q, t_kv)
    scores = tf.matmul(q_input, k_input, transpose_b=True)
    # (mbs, t_q, t_kv)
    distribution = tf.nn.softmax(scores)
    # (mbs, t_q, t_kv) x (mbs, t_kv, dim) -> (mbs, t_q, dim)
    output = tf.matmul(distribution, v_input)

    model = Model(inputs=[q_input, v_input, k_input], outputs=output)
    return  model([q, v, k])

def test_attention_3():
    scores = np.zeros((mbs, t_q, t_kv))
    for mb_idx in range(mbs):
        for q_idx in range(t_q):
            for kv_idx in range(t_kv):
                score = np.dot(q[mb_idx, q_idx], k[mb_idx, kv_idx])
                scores[mb_idx, q_idx, kv_idx] = score
    
    exp_scores_sum = np.zeros((mbs, t_q))
    for mb_idx in range(mbs):
        for q_idx in range(t_q):
            for kv_idx in range(t_kv):
                exp_scores_sum[mb_idx, q_idx] += np.exp(scores[mb_idx, q_idx, kv_idx])

    distribution = np.zeros((mbs, t_q, t_kv))
    for mb_idx in range(mbs):
        for q_idx in range(t_q):
            exp_score_sum = exp_scores_sum[mb_idx, q_idx]
            for kv_idx in range(t_kv):
                distribution[mb_idx, q_idx, kv_idx] = np.exp(scores[mb_idx, q_idx, kv_idx]) / exp_score_sum

    output = np.zeros((mbs, t_q, dim))
    for mb_idx in range(mbs):
        for q_idx in range(t_q):
            for dim_idx in range(dim):
                output[mb_idx, q_idx, dim_idx] = np.dot(distribution[mb_idx, q_idx], v[mb_idx, :, dim_idx])

    return output

def embedding_test():
    # Variable-length int sequences.
    q_input = tf.keras.Input(shape=(None,), dtype='int32', name='q_input')
    v_input = tf.keras.Input(shape=(None,), dtype='int32', name='v_input')

    max_tokens = 1000
    dimension = 10
    token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)

    q_emb = token_embedding(q_input)
    v_emb = token_embedding(v_input)

    nh = 5
    q_dense_layer = tf.keras.layers.Dense(nh, name='q_dense')
    v_dense_layer = tf.keras.layers.Dense(nh, name='v_dense')

    q_af = q_dense_layer(q_emb)
    v_af = v_dense_layer(v_emb)

    attention_layer = tf.keras.layers.Attention(name='attention')

    qv_att = attention_layer([q_emb, v_emb])

    model = Model(inputs=[q_input, v_input], outputs=qv_att)

    return model



def build_model():
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    value_input = tf.keras.Input(shape=(None,), dtype='int32')

    max_tokens = 1000
    dimension = 10

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)

    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        # Use 'same' padding so outputs have the same shape as inputs.
        padding='same')

    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])
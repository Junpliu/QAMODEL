# import tensorflow as tf
# import numpy as np
#
# a = np.array([[0.3, 0.5, 0.1],
#               [0.1, 0.3, 0.7],
#               [0.4, 0.1, 0.3],
#                [0.6, 0.1, 0.3]])
# b = np.array([[0.3, 0.6, 0.1],
#               [0.1, 0.3, 0.7],
#               [0.4, 0.1, 0.3],
#                [0.6, 0.1, 0.3]])
# print(a.shape)
# a = tf.convert_to_tensor(a)
# b = tf.convert_to_tensor(b)
#
# # max_pooling = tf.layers.MaxPooling1D(pool_size=4, strides=1, padding="valid")
# # b = max_pooling(a)
# c = tf.abs(tf.subtract(a, b))
# #
# with tf.Session() as sess:
#     print(sess.run(c))
#     print(c.shape)
# import tensorflow as tf
#
#
# def func():
#     # with tf.name_scope('a'):
#     #     var_1 = tf.Variable(initial_value=[0], name='var_1')
#     #     var_2 = tf.get_variable(name='var_2', shape=[1, ])
#     # with tf.variable_scope('b', reuse=tf.AUTO_REUSE):
#
#     # var_3 = tf.Variable(initial_value=[0], name='var_3')
#     with tf.variable_scope("c", reuse=tf.AUTO_REUSE):
#         filter_size = 2
#         embedding_size = 300
#         filter_num = 100
#         filter_shape = [filter_size, embedding_size, 1, filter_num]
#         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#         b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
#         ww = tf.get_variable("ww", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
#         bb = tf.get_variable("bb", shape=[filter_num], initializer=tf.constant_initializer(0.1))
#         tf.truncated_normal_initializer()
#     # print(var_1.name)
#     # print(var_2.name)
#     # print(var_3.name)
#     print(W.name)
#     print(b.name)
#     print(ww.name)
#     print(bb.name)
#     return W, b, ww, bb
#
# a, b, c, d = func()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     aa,bb, cc,dd = sess.run([a,b,c,d])
#     print(a, a.shape)
#     print(b, b.shape)
#     print(c, c.shape)
#     print(d, d.shape)


# with tf.variable_scope("a"):
#     func()
# with tf.variable_scope("b"):
#     func()

# import numpy as np
#
# np.random.seed(1213)
#
# embedding_dim = 3
# size_vocab = 5
# embedding_pad = np.zeros([1, embedding_dim], dtype=np.float32)
# embedding_matrix = np.random.uniform(-1, 1, [size_vocab - 1, embedding_dim]).astype(np.float32)
# embedding_matrix = np.concatenate([embedding_pad, embedding_matrix])
# print(embedding_matrix)
# print(embedding_matrix[0])
# print(type(embedding_matrix))
# a = [[1, 2, 3 , 4, 0], [2,3, 4, 0, 0]]
# l = [[4], [5]]
#
#
# import tensorflow as tf
#
# embed_const = tf.expand_dims(tf.constant(embedding_matrix[0], dtype=tf.float32), 0)
# # embed_matrix = np.random.uniform(-1,1, [size_vocab - 1, embedding_dim])
# embed_var = tf.Variable(embedding_matrix[1:], dtype=tf.float32, name = "embed_var")
# embeddings = tf.concat([embed_const, embed_var], axis = 0, name = 'embed')
# embed_seq = tf.nn.embedding_lookup(embeddings, a)
# mean = tf.div(tf.reduce_sum(embed_seq, axis=1), l)
# mean1 = tf.reduce_sum(embed_seq, axis=1)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a, b = sess.run([embeddings, mean])
#     print(a)
#     print("b", b)
#
#
# np.random.uniform()

# import tensorflow as tf
# import numpy as np
#
# document = tf.Variable(np.array([[[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]]))
# query = tf.Variable(np.array([[[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]]))
# affinity = tf.einsum("ndh,nqh->ndq", document, query)
# sequence_length = tf.Variable(np.array([1]))
# score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
# score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(affinity)[2]))
# affinity_mask_values = float('-inf') * tf.ones_like(affinity)
# res = tf.where(score_mask, affinity, affinity_mask_values)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a, b, c = sess.run([score_mask, affinity, res])
#     print(a.shape)
#     print(b.shape)
#     print(a)
#     print(b)
#     print(c)
# import tensorflow as tf
# import numpy as np
#
# initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=1221)
# tf.get_variable_scope().set_initializer(initializer)
#
# with tf.variable_scope("s"):
#     a = tf.get_variable("a", shape=[2, 4, 3], dtype=tf.float64)
#     b = tf.get_variable("b", shape=[2, 3], dtype=tf.float64)
#     wy = tf.get_variable("wy", shape=[3, 3], dtype=tf.float64)
#     wh = tf.get_variable("wh", shape=[3, 3], dtype=tf.float64)
#     b_ = tf.expand_dims(b, 1)
#     b_ = tf.tile(b_, [1, 4, 1])
#     wy_y = tf.matmul(tf.reshape(a, shape=[2*4, 3]), wy)
#     wh_h = tf.matmul(tf.reshape(b_, shape=[2*4, 3]), wh)
#     M = tf.nn.tanh(tf.add(wy_y, wh_h))
#     w = tf.get_variable("w", shape=[3, 1], dtype=tf.float64)
#     w_M = tf.matmul(M, w)
#     w_M_reshape = tf.transpose(tf.reshape(w_M, [-1, 4, 1]), [0, 2, 1])
#     alpha = tf.nn.softmax(w_M_reshape)
#     r = tf.matmul(alpha, a)
# # c = tf.einsum("nlk,lk->l", a, b)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     res = sess.run([b_, b, r])
#     print(res[0].shape, res[0])
#     print(res[1].shape, res[1])
#     print(res[2].shape, res[2])


# import tensorflow as tf
#
# a = tf.Variable([[0.122,0.2333,0.33333]])
# b = tf.Variable([[0.22,0.33,0.44]])
# c = tf.complex(a, b)
# W = tf.Variable
# d = tf.layers.dense(c, 6, kernel_initializer=tf.random_normal_initializer)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# res = sess.run(d)
# print(res)
# print(res.shape)


# import bisect
# #
# # a = [1,2,3,4,5]
# #
# # print(bisect.bisect_left(a, 7))
# # bisect.insort_left(a, 8)
# #
# # print(a)

# import re
#
# find = re.match("L(android|java).*->.*\(", "Landroidhhhh->fdfdf(")
# print(find.group())


# import tensorflow as tf
# import numpy as np
# import time
#
# tf.set_random_seed(1213)
# # np.random.seed(1213)
#
# m = 5
# n = 6
# num_filters = 2
# x_len = np.array([2, 3])
# y_len = np.array([4, 5])
# k1 = 3
# De = 2
# batch_size = 2
# x = np.array([[[2, 3], [3, 4], [0, 0], [0, 0], [0, 0]],
#               [[2, 3], [3, 4], [4, 5], [6, 7], [0, 0]]])
# y = np.array([[[11, 12], [13, 14], [15, 16], [17, 18], [0, 0], [0, 0]],
#               [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [0, 0]]])
# a = tf.convert_to_tensor(x, dtype=tf.float32)
# b = tf.convert_to_tensor(y, dtype=tf.float32)
# a_len = tf.convert_to_tensor(x_len, dtype=tf.int32)
# b_len = tf.convert_to_tensor(y_len, dtype=tf.int32)
#
# a_mask = tf.cast(tf.sequence_mask(a_len, maxlen=m), dtype=tf.int32)
# b_mask = tf.cast(tf.sequence_mask(b_len, maxlen=n), dtype=tf.int32)
# ab_mask = tf.cast(tf.einsum('bm,bn->bmn', a_mask, b_mask), dtype=tf.float32)
# ab_mask_tile = tf.tile(tf.expand_dims(ab_mask, -1), [1, 1, 1, num_filters])
# # Layer-1  #TODO: 训练太慢 换成矩阵操作
# con1ds = []
# for i in range(m-k1+1):
#     ai_ = []
#     for j in range(n-k1+1):
#         # ai = a[:, i:i+k1]
#         # bi = b[:, j:j+k1]
#
#         ai = tf.slice(a, [0, i, 0], [2, k1, De])
#         bi = tf.slice(b, [0, j, 0], [2, k1, De])
#         zij = tf.concat([ai, bi], axis=1, name="zij")
#         con1d = tf.layers.conv1d(inputs=zij,
#                                  filters=num_filters,
#                                  kernel_size=(2*k1),
#                                  strides=1,
#                                  padding="valid",
#                                  reuse=tf.AUTO_REUSE,
#                                  name="1-con1d")  # [batch_size, 1, num_filters]
#         gz_mask = tf.slice(ab_mask, [0, i, j], [2, k1, k1])
#         gz_mask = tf.reshape(gz_mask, [2, -1])
#         gz_mask = tf.cast(tf.reduce_any(gz_mask, axis=1), tf.float32)  # [batch_size]
#         gz_mask = tf.expand_dims(gz_mask, -1)
#         gz_mask = tf.expand_dims(gz_mask, -1)
#         gz_mask = tf.tile(gz_mask, [1, 1, num_filters])
#         zij_ = tf.multiply(gz_mask, con1d)
#         ai_.append(zij_)
#     ai_ = tf.concat(ai_, axis=1)
#     ai_ = tf.expand_dims(ai_, 1)
#     con1ds.append(ai_)
# con1ds = tf.concat(con1ds, axis=1)  # [batch_size, m-k1+1, n-k1+1, 100]
#
#
# # Layer-1  #TODO: 矩阵操作
# con1d1 = tf.layers.Conv1D(num_filters, k1, 1, 'same', name="1/con1d1")
# con1d2 = tf.layers.Conv1D(num_filters, k1, 1, 'same', name='1/con1d2')
# x_con1d = con1d1(a)
# y_con1d = con1d2(b)
# x_con1d_tile = tf.tile(tf.expand_dims(x_con1d, 2), [1, 1, n, 1])
# y_con1d_tile = tf.tile(tf.expand_dims(y_con1d, 1), [1, m, 1, 1])
# y_reshape = tf.reshape(y_con1d_tile, shape=[batch_size, m, n, num_filters])
# xy = tf.add(x_con1d_tile, y_reshape)
# xy_con1d = tf.multiply(ab_mask_tile, xy)
#
# # Layer-2
# pool2d = tf.layers.max_pooling2d(inputs=xy_con1d,
#                                  pool_size=(2, 2),
#                                  strides=2,
#                                  padding="same",
#                                  name="2/pool2d_1")
#
# # Layer-3
# con2d = tf.layers.conv2d(inputs=pool2d,
#                          filters=100,
#                          kernel_size=(2, 2),
#                          padding="same",
#                          name="3/con2d_1")
# # Layer-4-8
# pool2d_2 = tf.layers.max_pooling2d(con2d, (2, 2), 2, "same", name="4/pool2d_2")
# con2d_2 = tf.layers.conv2d(pool2d_2, 100, (2, 2), 1, "same", name="5/con2d_2")
# pool2d_3 = tf.layers.max_pooling2d(con2d_2, (2, 2), 2, "same", name="6/pool2d_3")
# batch_size = pool2d_3.get_shape()[0]
# flatten = tf.layers.flatten(pool2d_3)
# fc_1 = tf.layers.dense(flatten, 50, tf.nn.relu, name="7/mlp_1")
# fc_2 = tf.layers.dense(fc_1, 2, name="8/mlp_2")
#
# params = tf.trainable_variables()
# for param in params:
#     print("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print("a", a.get_shape())
# print("b", b.get_shape())
# print("a_mask", sess.run(a_mask))
# print("b_mask", sess.run(b_mask))
# print("ab_mask_tile", ab_mask_tile.get_shape())
# print("pool2d", pool2d.get_shape())
# print("con2d", con2d.get_shape())
# print("pool2d_2", pool2d_2.get_shape())
# print("con2d_2", con2d_2.get_shape())
# print("pool2d_3", pool2d_3.get_shape())
# print("fc_1", fc_1.get_shape())

# import tensorflow as tf
#
# batch_size = 8
#
# v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
#
# a = tf.matmul(v1, v2)
# b = tf.multiply(v1, v2)
#
# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# print(sess.run(a))
# print(sess.run(b))


import numpy as np
import tensorflow as tf

labels = np.array([[1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 0]], dtype=np.uint8)
predictions = np.array([[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 1, 1, 1]], dtype=np.uint8)
n_batches = len(labels)

graph = tf.Graph()
with graph.as_default():
    # Placeholders to take in batches onf data
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
    tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])
    # Define the metric and update operations
    tf_metric, tf_metric_update = tf.metrics.accuracy(tf_label,
                                                      tf_prediction,
                                                      name="my_metric")
    tf.summary.scalar("acc", tf_metric)
    merged = tf.summary.merge_all()
    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    for i in range(n_batches):
        # Reset the running variables
        session.run(running_vars_initializer)
        # Update the running variables on new batch of samples
        feed_dict = {tf_label: labels[i], tf_prediction: predictions[i]}
        session.run(tf_metric_update, feed_dict=feed_dict)
        # Calculate the score on this batch
        summary, score = session.run([merged, tf_metric])
        print("[TF] batch {} score: {}".format(i, score))
        print(summary)

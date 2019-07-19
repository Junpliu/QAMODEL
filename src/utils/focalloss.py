import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(logits, labels, alpha=0.25, gamma=2):
  """Compute focal loss for predictions.
    Multi-labels Focal loss formula:
      FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
           ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target.
  Args:
   logits: A float tensor of shape [batch_size, num_classes] representing the predicted logits for each class
   labels: A float tensor of shape [batch_size, num_classes] representing one-hot encoded classification targets
   alpha: A scalar tensor for focal loss alpha hyper-parameter
   gamma: A scalar tensor for focal loss gamma hyper-parameter
  Returns:
    loss: A (scalar) tensor representing the value of the loss function
  """
  pred = tf.nn.softmax(logits)
  # labels = tf.one_hot(labels, depth=2)

  zeros = array_ops.zeros_like(pred, dtype=pred.dtype)

  cond = (labels > zeros)
  # For positive prediction, only need consider front part loss, back part is 0;
  # target > zeros <=> z=1, so positive coefficient = z - p.
  pos_p_sub = tf.where(cond, labels - pred, zeros)

  # For negative prediction, only need consider back part loss, front part is 0;
  # target > zeros <=> z=1, so negative coefficient = 0.
  neg_p_sub = tf.where(cond, zeros, pred)

  per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
  return tf.reduce_sum(per_entry_cross_ent, axis=1)


# def focal_loss_softmax(labels, logits, gamma=2):
#   """
#   Computer focal loss for multi classification
#   Args:
#     labels: A int32 tensor of shape [batch_size].
#     logits: A float32 tensor of shape [batch_size,num_classes].
#     gamma: A scalar for focal loss gamma hyper-parameter.
#   Returns:
#     A tensor of the same shape as `lables`
#   """
#   y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
#   labels = tf.one_hot(labels, depth=y_pred.shape[1])
#   L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
#   L = tf.reduce_sum(L, axis=1)
#   return L


# # y_pred = tf.Variable([1., 3., 5.])
# y_pred = tf.Variable([[1., 4.], [1., 3.], [1., 5.]])
# y_label = tf.Variable([0, 1, 1])
# y_true = tf.one_hot(y_label, depth=2)
#
# b = focal_loss(y_pred, y_true, gamma=2, alpha=0.25)
# # c = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
# # b = focal_loss_softmax(y_label, y_pred, gamma=2)
# loss = tf.reduce_mean(b)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# res = sess.run([loss])
# print(res[0])
# print(res[0].shape)

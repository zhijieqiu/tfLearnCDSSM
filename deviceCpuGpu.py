import tensorflow as tf

# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

#allow_growth option which attemps to allocate only as much GPU memory based on runtime allocations: it starts out allocating evry little memory, and as Sessions
#get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow
config = tf.ConfigProto()
config.gpu_options_allow_growth = True

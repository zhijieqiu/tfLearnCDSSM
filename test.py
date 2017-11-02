import tensorflow as tf
import numpy as np
import collections
print(collections.Counter([1,3,4,1]))
print(np.random.randn(10,4))
node1 = tf.constant(3.0,dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1,node2)

print('hello,world')

W = tf.Variable([0.3],dtype = tf.float32)
b = tf.Variable(-0.3, dtype = tf.float32)

x = tf.placeholder(dtype = tf.float32)
linear_model = W*x + b
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)
square_deltas = tf.square(linear_model - y)
loss = tf.reduce_mean(square_deltas)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})

curr_W, curr_b ,curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})

print('W: {} b: {} loss: {}'.format(curr_W,curr_b,curr_loss))



dataset = tf.contrib.data.Dataset.range(100)
a = []
#dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.map(lambda x : [x]*3)
dataset = dataset.batch(4)
print(dataset.output_shapes)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]

# Define training and validation datasets with the same structure.
training_dataset = tf.contrib.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.contrib.data.Dataset.range(50)
training_dataset = training_dataset.batch(3)
validation_dataset = validation_dataset.batch(2)
# my_iterator = validation_dataset.make_one_shot_iterator()
# my_next_element = my_iterator.get_next()
# print(sess.run(my_next_element))
# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
print(training_dataset.output_shapes)
iterator = tf.contrib.data.Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100//3):
    pass
    print(sess.run(next_element))
    print('-----------')

  # Initialize an iterator over the validation dataset.
  # sess.run(validation_init_op)
  # for _ in range(50):
  #   print(sess.run(next_element))
  #   print('&&&&&&&&&&&&&&&')
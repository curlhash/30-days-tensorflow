import tensorflow as tf
x1 = tf.constant(10)
x2 = tf.constant(20)
result = tf.multiply(x1, x2)
print(result)
with tf.Session() as sess:
	print(sess.run(result))


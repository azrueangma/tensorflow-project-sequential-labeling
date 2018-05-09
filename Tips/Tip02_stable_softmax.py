import tensorflow as tf

def stable_softmax(x, axis=None):
    z = tf.reduce_max(x, axis=axis, keepdims = True )
    numerator = tf.exp(tf.subtract(x,z))
    denominator = tf.reduce_sum(numerator, axis=axis, keepdims = True)
    softmax_result = tf.divide(numerator, denominator)
    return softmax_result

a = tf.constant([1.0, 3.0, 2.0, 1.5])

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(a)))
    print(sess.run(stable_softmax(a)))

    
'''
[0.07839411 0.57925844 0.21309727 0.12925003]
[0.07839411 0.57925844 0.21309729 0.12925005]
'''

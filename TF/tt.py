import tensorflow as tf

with tf.name_scope("name_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2, name="adds")

print(v1.name)
print(v2.name)
print(a.name)

with tf.variable_scope("variable_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2, name="adds")

print(v1.name)
print(v2.name)
print(a.name)


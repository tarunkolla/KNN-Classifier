import tensorflow as tf

#MNIST dataset from keras
mnist = tf.keras.datasets.mnist

#tuples that have image in x and value in y.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reduces to 0-1 from 0-255
#this may not have much of a change on the output
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Using sequential model with 2 layered neural network with each having 128 neurons
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer= 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)
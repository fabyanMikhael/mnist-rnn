import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout

(x_train, y_train) ,  (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0

optimizer = keras.optimizers.Adam(learning_rate=0.9e-2,decay=1e-5)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = keras.Sequential()
model.add(LSTM( 128, input_shape=(28,28) ))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test,y_test))
model.save("model.tf")
model.evaluate(x_test,y_test)
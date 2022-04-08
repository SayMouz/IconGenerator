import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

array = np.random.rand(600, 32, 32)

loaded_arr = np.loadtxt("data.csv")
loaded_arr = loaded_arr[:600]

X = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // array.shape[2], array.shape[2])
Y = np.loadtxt("tags_data.csv")
Y = Y[:600]

X_shuffle, Y_shuffle = unison_shuffled_copies(X, Y)
x_train = X_shuffle[:500]
y_train = Y_shuffle[:500]

x_test = X_shuffle[500:]
y_test = Y_shuffle[500:]


num_classes = 23
input_shape = (32, 32, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),  # -> 28x28x1
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),  # -> 28x28x16
        layers.MaxPooling2D(pool_size=(2, 2)),  # -> 6x6x32
        layers.Conv2D(64, kernel_size=(3, 3),
                      kernel_initializer="lecun_normal",
                      activation="selu", padding="same"),  # -> 6x6x64
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # -> 3x3x64
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", strides=(3, 3)),  # -> 1x1x64
        layers.Flatten(),  # -> 64
        layers.Dropout(0.5),  # -> 64
        layers.Dense(num_classes, activation="sigmoid"),  # -> 10
    ]
)

model.summary()

batch_size = 20
epochs = 100

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# make a prediction on the test set
yhat = model.predict(x_test)
# round probabilities to class labels
yhat = yhat.round()
# calculate accuracy
acc = accuracy_score(y_test, yhat)
# store result
print('>%.3f' % acc)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

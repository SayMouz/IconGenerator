import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


img_size = 32
# -------- Data loading --------------- #
print('--- Loading data ...')
# Load icon data from file
loaded_ic_data = np.loadtxt("./ic_data/1000icons_x_5tags/ic_data_clean.csv")
# Reshape to the correct img shape
X = loaded_ic_data.reshape(
        loaded_ic_data.shape[0], loaded_ic_data.shape[1] // img_size, img_size)
# Load tags data from file
Y = np.loadtxt("./ic_data/1000icons_x_5tags/tags_data_clean.csv")
print(f'--- Data loaded : {len(X)} samples ---')
# ------------------------------------- #

# ------ Create train and test datasets -----#
train_test_ratio = 0.8

print("--- Creating train and test datasets ---")
train_size = int(train_test_ratio*len(X))
X_shuffle, Y_shuffle = unison_shuffled_copies(X, Y)
x_train = X_shuffle[:train_size]
y_train = Y_shuffle[:train_size]

x_test = X_shuffle[train_size:]
y_test = Y_shuffle[train_size:]
print("--- Train and test datasets created ---")
print(f"--- Train dataset : {len(x_train)} samples | Test dataset : {len(x_test)} samples ---")
# -----------------------------------------#

# --------- Create model --------------- #
num_classes = np.shape(y_train)[1]
input_shape = (img_size, img_size, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3),
                      kernel_initializer="lecun_normal",
                      activation="selu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", strides=(3, 3)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid"),
    ]
)
model.summary()
# -------------------------------------------------- #

# ---------------- Train --------------------------#
batch_size = 100
epochs = 100

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# ------------------------------------------------- #

# ------------- Test ------------------------------ #
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
# ----------------------------------------------- #
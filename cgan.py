import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers, models

"""
Source for the major part of the code :
https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
"""

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# --- Discriminator definition --- #
def define_discriminator(img_size, n_classes):
    # Label input
    label_input = layers.Input(shape=(n_classes,))
    li = layers.Dense(img_size * img_size)(label_input)
    li = layers.Reshape((img_size, img_size, 1))(li)

    # Image input
    image_input = layers.Input(shape=(img_size, img_size, 1))

    # Concatenation
    merge = layers.Concatenate()([image_input, li])

    # Discriminator inside
    fe = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.GlobalMaxPooling2D()(fe)

    # output
    d_output_layer = layers.Dense(1)(fe)

    # Model
    discriminator = models.Model([image_input, label_input], d_output_layer)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return discriminator


# -------------------------------- #

# ---- Generator definition ------ #

def define_generator(latent_dim, n_classes=5):
    # Label input
    label_input = layers.Input(shape=(n_classes,))
    li = layers.Dense(8 * 8)(label_input)
    li = layers.Reshape((8, 8, 1))(li)

    # Image generator input
    in_lat = layers.Input(shape=(latent_dim,))
    n_nodes = 128 * 8 * 8
    gen = layers.Dense(n_nodes)(in_lat)
    gen = layers.LeakyReLU(alpha=0.4)(gen)
    gen = layers.Reshape((8, 8, 128))(gen)

    # Concatenation
    merge = layers.Concatenate()([gen, li])

    # Upsamples
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = layers.LeakyReLU(alpha=0.4)(gen)
    gen = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.4)(gen)

    # Output
    out_layer = layers.Conv2D(1, (8, 8), activation='sigmoid', padding='same')(gen)

    # Model
    generator = models.Model([in_lat, label_input], out_layer)
    return generator


# -------------------------------- #

# ----- CGAN definition ---------- #

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = models.Model([gen_noise, gen_label], gan_output)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# -------------------------------- #

# ---------- Train --------------- #

# # select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=5):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save('cgan_generator3.h5')


# -------------------------------- #


image_size = 32
# -------- Data loading --------------- #
print('--- Loading data ...')
# Load icon data from file
loaded_ic_data = np.loadtxt("./ic_data/1000icons_x_5tags/ic_data_clean.csv")
# Reshape to the correct img shape
X = loaded_ic_data.reshape(
    loaded_ic_data.shape[0], loaded_ic_data.shape[1] // image_size, image_size)
# Load tags data from file
Y = np.loadtxt("./ic_data/1000icons_x_5tags/tags_data_clean.csv")
print(f'--- Data loaded : {len(X)} samples ---')

num_classes = np.shape(Y)[1]
# ------------------------------------- #
"""
# --- Show first 100 images --- #
for i in range(100):
    # define subplot
    pyplot.subplot(10, 10, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    pyplot.imshow(X[i], cmap='gray_r')
pyplot.show()
# ------------------------------ #
"""

# ---- Dataset creation -------- #
X = np.expand_dims(X, axis=-1)
dataset = [X, Y]
# ------------------------------ #

d_m = define_discriminator(image_size, 5)
g_m = define_generator(100, 5)
d_m.summary()
g_m.summary()

cgan = define_gan(g_m, d_m)

train(g_m, d_m, cgan, dataset, 100, 50, 128)

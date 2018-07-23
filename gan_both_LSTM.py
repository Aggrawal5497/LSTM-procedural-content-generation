from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.layers import LSTM, UpSampling1D, TimeDistributed
from keras.optimizers import Adam
import noise
import matplotlib.pyplot as plt

import sys

import numpy as np

class LSTMGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        #self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = 28


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(28,28,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):

        model = Sequential()
        model.add(LSTM(512, return_sequences = True, input_dim = 28))
        model.add(Dropout(0.8))
        model.add(LSTM(512, return_sequences = True))
        model.add(Dropout(0.8))
        model.add(TimeDistributed(Dense(28)))

        model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(LSTM(128, input_shape=(28, 28,)))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        return model

    def train(self, epochs, n, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()
        print(X_train.shape)
        
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)
        print(X_train.shape)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            gen_imgs = self.generator.predict(n)

            # Train the discriminator (real classified as ones and generated as zeros)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(n, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images_mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = LSTMGAN()
    w = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            w[i, j] = noise.pnoise2(i/100, j/100, 6)
    no = np.zeros((50, 28, 28))
    for i in range(50):
        no[i, :, :] = w
    dcgan.train(epochs=5000, batch_size=50, save_interval=50, n = no)

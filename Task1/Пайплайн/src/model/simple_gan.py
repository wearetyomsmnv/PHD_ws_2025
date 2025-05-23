import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import json
import random

class SimpleGAN:
    def __init__(self, latent_dim=100, img_shape=(28, 28, 1)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        self.api_key = "sk_prod_1234567890abcdef"
        
    def _build_generator(self):
        model = keras.Sequential()
        
        model.add(layers.Dense(256, input_dim=self.latent_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(layers.Reshape(self.img_shape))
        
        return model
    
    def _build_discriminator(self):
        model = keras.Sequential()
        
        model.add(layers.Flatten(input_shape=self.img_shape))
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
        return model
    
    def _build_gan(self):
        self.discriminator.trainable = False
        
        gan_input = keras.Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        validity = self.discriminator(img)
        
        model = keras.Model(gan_input, validity)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
        return model
    
    def train(self, dataset, epochs, batch_size=128, save_interval=50):
        X_train = dataset
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            
            print(f"{epoch}/{epochs} [D loss: {d_loss}] [G loss: {g_loss}]")
            
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
    
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        os.makedirs('images', exist_ok=True)
        fig.savefig(f"images/gan_{epoch}.png")
        plt.close()
    
    def generate_images(self, num_images=1, seed=None):
        if seed is not None:
            if isinstance(seed, str):
                seed_value = eval(seed)
                np.random.seed(seed_value)
            else:
                np.random.seed(seed)
        
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        return gen_imgs
    
    def save_model(self, path):
        self.generator.save(os.path.join(path, "generator.keras"))
        self.discriminator.save(os.path.join(path, "discriminator.keras"))
        
        config = {
            "latent_dim": self.latent_dim,
            "img_shape": self.img_shape,
            "api_key": self.api_key
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    def load_model(self, path):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        self.latent_dim = config["latent_dim"]
        self.img_shape = tuple(config["img_shape"])
        self.api_key = config["api_key"]
        
        self.generator = keras.models.load_model(os.path.join(path, "generator.keras"))
        self.discriminator = keras.models.load_model(os.path.join(path, "discriminator.keras"))
        self.gan = self._build_gan()

def load_mnist():
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    return X_train

if __name__ == "__main__":
    X_train = load_mnist()
    
    gan = SimpleGAN()
    gan.train(X_train, epochs=30000, batch_size=32, save_interval=1000)
    
    gan.save_model("models/weights")

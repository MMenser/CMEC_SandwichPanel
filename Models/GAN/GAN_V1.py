# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from matplotlib.image import imread

def preprocess_data(txt_folder, image_folder, test_size=0.15):
    file_mapping = {} 
    
    txt_files = sorted([f for f in os.listdir(txt_folder) if f.endswith('.txt')])
    print(f"Found {len(txt_files)} text files in {txt_folder}: {txt_files}")  
    
    identifiers = []  
    
    for filename in txt_files:
        match = re.search(r'(\d+_\d+_\d+)(?=\.\w+$)', filename)
        
        if match:
            identifier = match.group(1)
            identifiers.append(identifier)  # Add the identifier to the list
            
            txt_path = os.path.join(txt_folder, filename)
            
            try:
                data = pd.read_csv(txt_path, delimiter='\t', header=None).values.flatten()
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            strain = data[::2] 
            stress = data[1::2] 
            data_normalized = np.column_stack((strain, stress)) 
            
            image_filename = f"vf_{identifier}.jpg"
            image_path = os.path.join(image_folder, image_filename)
            
            if os.path.exists(image_path):
                try:
                    image = imread(image_path)
                    
                    if len(image.shape) == 3:  
                        image = np.mean(image, axis=-1)

                    file_mapping[identifier] = {
                        'data': data_normalized,
                        'image': image
                    }
                    
                except Exception as e:
                    print(f"Error loading .jpg file {image_filename}: {e}")
            else:
                print(f"Warning: Image file {image_filename} not found.")

    print(f"Total valid identifiers found: {len(file_mapping)}")  

    if len(file_mapping) == 0:
        raise ValueError("No valid data was found to split.")

    train_ids, val_ids = train_test_split(identifiers, test_size=test_size, random_state=42)
    
    train_data = [file_mapping[id]['data'] for id in train_ids]
    train_images = [file_mapping[id]['image'] for id in train_ids]
    val_data = [file_mapping[id]['data'] for id in val_ids]
    val_images = [file_mapping[id]['image'] for id in val_ids]
    
    return np.array(train_data), np.array(train_images), np.array(val_data), np.array(val_images), identifiers, file_mapping


def display_data_and_image(identifier, file_mapping):
    if identifier not in file_mapping:
        print(f"Identifier '{identifier}' not found in file_mapping.")
        return
    
    image = file_mapping[identifier]['image']  
    data = file_mapping[identifier]['data'] 

    if image is None or data is None:
        print("Error: Image or data is missing!")
        return
    
    data = np.array(data).reshape(21, 2)  
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    
    strain = sorted_data[:, 0]
    stress = sorted_data[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(image, cmap='gray', interpolation='nearest')
    ax[0].set_title(f"Microstructure Image: {identifier}")
    ax[0].axis('off')

    ax[1].plot(strain, stress, marker='o', linestyle='-', color='b', label="Stress-Strain")
    ax[1].set_xlabel("Strain")
    ax[1].set_ylabel("Stress")
    ax[1].set_title("Stress-Strain Curve")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


txt_folder = '/home/cimpi/Desktop/Tiana/MicroGen/result2'  
image_folder = '/home/cimpi/Desktop/Tiana/MicroGen/microstructure_gen2'

train_data, train_images, val_data, val_images, identifiers, file_mapping = preprocess_data(txt_folder, image_folder)

identifier = '3_10_2' 
display_data_and_image(identifier, file_mapping)



# %%
import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, input_dim, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(512, activation='relu')
        self.dense3 = layers.Dense(img_size * img_size, activation='sigmoid')  # Output image
        self.volume_fc = layers.Dense(1)  # Volume fraction output

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        volume_fraction = self.volume_fc(x)  # Extract volume fraction
        img = self.dense3(x)  # Continue processing for image
        return tf.reshape(img, (-1, self.img_size, self.img_size)), volume_fraction


class Discriminator(tf.keras.Model):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.dense1 = layers.Dense(512, activation='leaky_relu')
        self.dense2 = layers.Dense(256, activation='leaky_relu')
        self.dense3 = layers.Dense(1, activation='sigmoid')  # Probability of being real

    def call(self, img, volume_fraction):
        img_flat = self.flatten(img)
        volume_fraction = tf.reshape(volume_fraction, (-1, 1))  
        x = self.concat([img_flat, volume_fraction])  # Concatenate volume fraction
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


# %%
import tensorflow as tf
import numpy as np

z_dim = 100  # Latent space dimension for Generator input

# Create the Generator and Discriminator models
generator = Generator(input_dim=2, img_size=128)  
discriminator = Discriminator(img_size=128)

# Optimizers for Generator and Discriminator (Defined once, outside the train_step function)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)

# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Function to calculate generator and discriminator losses
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) 

def extract_volume_fraction(identifier):
    # Assuming identifier is in the format 'X_Y_Z', where Y is the volume fraction
    parts = identifier.split('_')
    return int(parts[1])  # The middle number (Y) is the volume fraction

# Define the train_step function
latent_dim = 100

@tf.function
def train_step(real_data, real_images, identifiers, generator, discriminator):
    # Create noise for the generator input
    noise = tf.random.normal([real_images.shape[0], latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Get the generator output
        fake_image, fake_volume_fraction = generator(noise, training=True)
        
        # Get the discriminator's output for both real and fake images
        real_volume_fraction = [extract_volume_fraction(identifier) for identifier in identifiers]
        real_volume_fraction = tf.convert_to_tensor(real_volume_fraction, dtype=tf.float32)
        
        real_output = discriminator(real_images, real_volume_fraction, training=True)
        fake_output = discriminator(fake_image, fake_volume_fraction, training=True)
        
        # Calculate losses
        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
    
    # Compute gradients and apply them
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply the gradients using the optimizer
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train_gan(train_data, train_images, identifiers, epochs=100, batch_size=32):
    for epoch in range(epochs):
        idx = np.random.randint(0, len(train_data), batch_size)
        real_data = train_data[idx]  # Get a batch of stress-strain data
        real_images = train_images[idx]  # Get the corresponding batch of images
        real_identifiers = [identifiers[i] for i in idx]  # Get the corresponding batch of identifiers
        
        gen_loss, disc_loss = train_step(real_data, real_images, real_identifiers, generator, discriminator)
        
        print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {gen_loss.numpy():.4f}, Discriminator Loss: {disc_loss.numpy():.4f}")

    generator.save("generator_model.h5")
    discriminator.save("discriminator_model.h5")
    print("Models saved!")

train_gan(train_data, train_images, identifiers, epochs=100, batch_size=32)


# %%
import tensorflow as tf
import numpy as np
import sys

class GANTrainer:
    def __init__(self, generator, discriminator, input_dim, img_size, batch_size=32, epochs=10000, lr=0.0002):
        self.generator = generator
        self.discriminator = discriminator
        self.input_dim = input_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.gen_optimizer = tf.keras.optimizers.Adam(lr)
        self.disc_optimizer = tf.keras.optimizers.Adam(lr)
        
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def train_step(self, real_data, real_images, real_vf):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.input_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images, fake_vf = self.generator(noise, training=True)
            
            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))
            
            real_predictions = self.discriminator(real_images, real_vf, training=True)
            fake_predictions = self.discriminator(fake_images, fake_vf, training=True)
            
            disc_loss_real = self.loss_fn(real_labels, real_predictions)
            disc_loss_fake = self.loss_fn(fake_labels, fake_predictions)
            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            
            gen_loss = self.loss_fn(real_labels, fake_predictions)
        
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def train(self, train_data, train_images, train_vf):
        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_images, train_vf))
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)
        
        for epoch in range(self.epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            batch_count = 0
            
            for real_data, real_images, real_vf in dataset:
                gen_loss, disc_loss = self.train_step(real_data, real_images, real_vf)
                epoch_gen_loss += gen_loss.numpy()
                epoch_disc_loss += disc_loss.numpy()
                batch_count += 1
            
            epoch_gen_loss /= batch_count
            epoch_disc_loss /= batch_count
            
            print(f"Epoch {epoch + 1}/{self.epochs}, Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}", flush = True)
            sys.stdout.flush()
        print("Training complete!")

# %%




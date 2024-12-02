import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.manifold import TSNE

# Define the sampling function for the VAE latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Define a custom loss function using Keras' built-in loss mechanism
def vae_loss(spike_input_true, spike_output_pred, behavior_input_true, behavior_output_pred, z_mean, z_log_var):
    # Reconstruction loss for spiking activity (binary crossentropy)
    spike_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(spike_input_true, spike_output_pred))
    # Reconstruction loss for behavior data (mean squared error)
    behavior_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(behavior_input_true, behavior_output_pred))
    # KL divergence (encourages latent space to approximate a standard Gaussian)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return spike_loss + behavior_loss + kl_loss

# Function to create and train the VAE model
def train_vae(X_train, y_train, X_valid, y_valid, X_test, y_test, latent_dim=10, epochs=100, batch_size=64):
    # Input dimensions
    spike_input_shape = X_train.shape[1:]  # (520,) for flattened spiking data
    behavior_input_shape = y_train.shape[1:]  # (2,) for eye position data

    # Encoder
    spike_input = layers.Input(shape=spike_input_shape, name='spike_input')
    behavior_input = layers.Input(shape=behavior_input_shape, name='behavior_input')

    # Concatenate spiking and behavior inputs
    concat_inputs = layers.Concatenate()([spike_input, behavior_input])

    # Dense layers to learn latent variables
    x = layers.Dense(256, activation='relu')(concat_inputs)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # Latent space sampling
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder for spiking activity
    decoder_spike = layers.Dense(128, activation='relu')(z)
    decoder_spike = layers.Dense(256, activation='relu')(decoder_spike)
    decoder_spike_output = layers.Dense(520, activation='sigmoid', name='decoder_spike_output')(decoder_spike)

    # Decoder for behavior (eye position)
    decoder_behavior = layers.Dense(32, activation='relu')(z)
    decoder_behavior_output = layers.Dense(2, name='decoder_behavior_output')(decoder_behavior)

    # Define the VAE model
    vae = models.Model(inputs=[spike_input, behavior_input], outputs=[decoder_spike_output, decoder_behavior_output])

    # Compile the VAE model using custom loss
    def combined_loss(y_true, y_pred):
        spike_input_true, behavior_input_true = y_true
        spike_output_pred, behavior_output_pred = y_pred
        return vae_loss(spike_input_true, spike_output_pred, behavior_input_true, behavior_output_pred, z_mean, z_log_var)
    
    vae.compile(optimizer='adam', loss=combined_loss)

    # Train the VAE model
    history = vae.fit([X_train, y_train], [X_train, y_train],   # Inputs as lists
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=([X_valid, y_valid], [X_valid, y_valid]))

    # Get latent space representations for train, validation, and test sets
    encoder_model = models.Model(inputs=[spike_input, behavior_input], outputs=z)
    latent_train = encoder_model.predict([X_train, y_train])
    latent_valid = encoder_model.predict([X_valid, y_valid])
    latent_test = encoder_model.predict([X_test, y_test])

    # Get reconstructions (spiking activity and behavior)
    reconstructed_train = vae.predict([X_train, y_train])
    reconstructed_valid = vae.predict([X_valid, y_valid])
    reconstructed_test = vae.predict([X_test, y_test])

    # Reconstruction accuracy (MSE for behavior and spiking)
    def reconstruction_accuracy(real, reconstructed):
        return np.mean(np.square(real - reconstructed))

    train_reconstruction_accuracy = reconstruction_accuracy(y_train, reconstructed_train[1])
    valid_reconstruction_accuracy = reconstruction_accuracy(y_valid, reconstructed_valid[1])
    test_reconstruction_accuracy = reconstruction_accuracy(y_test, reconstructed_test[1])

    # Dimensionality reduction for latent space (using t-SNE for visualization)
    tsne = TSNE(n_components=2)
    latent_2d_train = tsne.fit_transform(latent_train)
    latent_2d_test = tsne.fit_transform(latent_test)

    # Return all data for plotting and interpretation
    return {
        'history': history,
        'latent_train': latent_train,
        'latent_valid': latent_valid,
        'latent_test': latent_test,
        'latent_2d_train': latent_2d_train,
        'latent_2d_test': latent_2d_test,
        'reconstructed_train': reconstructed_train,
        'reconstructed_valid': reconstructed_valid,
        'reconstructed_test': reconstructed_test,
        'train_reconstruction_accuracy': train_reconstruction_accuracy,
        'valid_reconstruction_accuracy': valid_reconstruction_accuracy,
        'test_reconstruction_accuracy': test_reconstruction_accuracy,
    }

# Example usage:

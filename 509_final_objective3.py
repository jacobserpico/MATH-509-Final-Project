import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, plot, xlim, ylim
from ipywidgets import interact

class Particle_Tracking_Training_Data(tf.Module):
    def __init__(self, Nt, rings=True):
        self.Nt = int(Nt)
        self.Ny = self.Nx = 256
        self.d = 3
        ximg = [[[i, j] for i in np.arange(self.Ny)]
                for j in np.arange(self.Nx)]
        self.ximg = np.float32(ximg)

        x = np.arange(self.Nx) - self.Nx//2
        y = np.arange(self.Ny) - self.Ny//2
        X0, Y0 = np.meshgrid(x, y)
        self.X = np.float32(X0)
        self.Y = np.float32(Y0)

        if rings:
            self.ring_indicator = 1.
        else:
            self.ring_indicator = 0.

        self._gen_video = tf.function(
            input_signature=(
                tf.TensorSpec(shape=[self.Ny, self.Nx, self.Nt, None], dtype=tf.float32),
                tf.TensorSpec(shape=[self.Nt, None], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
        )(self._gen_video)

        self._gen_labels = tf.function(
            input_signature=(
                tf.TensorSpec(shape=[self.Ny, self.Nx, self.Nt, None], dtype=tf.float32),)
        )(self._gen_labels)

    def __call__(self, kappa, a, IbackLevel, Nparticles, sigma_motion):
        xi = self._sample_motion(Nparticles, sigma_motion)
        XALL = (self.ximg[:, :, None, None, :] - xi[None, None, :, :, :2])
        r = tf.math.sqrt(XALL[..., 0]**2 + XALL[..., 1]**2)
        z = xi[..., 2]
        I = self._gen_video(r, z, kappa, a, IbackLevel)
        labels = self._gen_labels(r)
        return I, labels, xi

    @staticmethod
    def rand(n):
        return tf.random.uniform([n], dtype=tf.float32)

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),))
    def _sample_motion(self, Nparticles, sigma_motion):
        b_lower = tf.constant([-10, -10, -30.], tf.float32)
        b_upper = tf.constant([self.Nx+10, self.Ny+10, 30.], tf.float32)
        U = tf.random.uniform([1, Nparticles, self.d], dtype=tf.float32)
        X0 = b_lower + (b_upper - b_lower) * U
        dX = tf.random.normal([self.Nt, Nparticles, self.d], stddev=sigma_motion, dtype=tf.float32)
        X = X0 + tf.math.cumsum(dX, axis=0)
        X = tf.math.abs(X - b_lower) + b_lower
        X = -tf.math.abs(b_upper - X) + b_upper
        return X

    def _gen_video(self, r, z, kappa, a, IbackLevel):
        uw = (0.5 + self.rand(1)) / 2.
        un = tf.floor(3 * self.rand(1))
        uampRing = 0.2 + 0.8 * self.rand(1)
        ufade = 15 + 10 * self.rand(1)
        ufadeMax = 0.85
        fade = (1. - ufadeMax * tf.abs(tf.tanh(z / ufade)))
        core = tf.exp(-(r ** 2 / (8. * a)) ** 2)
        ring = fade * (tf.exp(-(r - z) ** 4 / (a ** 4)) + 0.5 * uampRing * tf.cast(r < z, tf.float32))
        I = tf.transpose(tf.reduce_sum(fade * (core + self.ring_indicator * ring), axis=3), [2, 0, 1])
        I += IbackLevel * tf.sin(
            self.rand(1) * 6 * tf.constant(np.pi, dtype=tf.float32) / 512 *
            tf.sqrt(self.rand(1) * (self.X - self.rand(1) * 512) ** 2 +
                    self.rand(1) * (self.Y - self.rand(1) * 512) ** 2))
        I += tf.random.normal([self.Nt, self.Ny, self.Nx], stddev=kappa, dtype=tf.float32)
        Imin = tf.reduce_min(I)
        Imax = tf.reduce_max(I)
        I = (I - Imin) / (Imax - Imin)
        I = tf.round(I * tf.maximum(256., tf.round(2 ** 16 * self.rand(1))))
        return I

    def _gen_labels(self, r):
        R_detect = 3.
        detectors = tf.reduce_sum(tf.cast(r[::2, ::2, :, :] < R_detect, tf.int32), axis=3)
        P = tf.transpose(tf.cast(detectors > 0, tf.int32), [2, 0, 1])
        labels = tf.stack([1 - P, P], 3)
        return labels

# Main simulation settings
Nt = 50
kappa = 0.1
a = 3.
IbackLevel = 0.15
Nparticles = 10
sigma_motion = 2.3

# Initialize the particle tracking data generator
pt = Particle_Tracking_Training_Data(Nt)

# Generate a video
vid, labels, tracks = pt(kappa, a, IbackLevel, Nparticles, sigma_motion)

# Function to extract patches around each particle's coordinates
def extract_patches_tf(vid, tracks, patch_size):
    half_patch = patch_size // 2
    Nt, Ny, Nx = vid.shape
    _, Nparticles, _ = tracks.shape
    patches = []
    z_coords = []

    for t in range(Nt):
        for p in range(Nparticles):
            x, y, z = tracks[t, p, :].numpy()  # Convert tensor to numpy array
            x = int(max(min(round(x), Nx - 1), 0))
            y = int(max(min(round(y), Ny - 1), 0))
            x_min = max(x - half_patch, 0)
            y_min = max(y - half_patch, 0)
            x_max = min(x + half_patch + 1, Nx)
            y_max = min(y + half_patch + 1, Ny)
            patch = vid[t, y_min:y_max, x_min:x_max].numpy()  # Convert tensor to numpy array
            pad_y = patch_size - patch.shape[0]
            pad_x = patch_size - patch.shape[1]
            patch = np.pad(patch, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)
            patches.append(patch)
            z_coords.append(z)

    return np.array(patches), np.array(z_coords)

patch_size = 51
vid = tf.convert_to_tensor(vid, dtype=tf.float32)
tracks = tf.convert_to_tensor(tracks, dtype=tf.float32)
patches, z_coords = extract_patches_tf(vid, tracks, patch_size)

# Model to estimate z-position
def create_model(input_shape=(51, 51, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model_Z_Position = create_model()
patches = np.expand_dims(patches, axis=-1)  # Ensure correct shape for CNN input

# Training
history = model_Z_Position.fit(patches, z_coords, epochs=10, validation_split=0.2)

# Visualize one frame with particle tracks
@interact(t=(0, Nt-1, 1))
def plotfn(t=0, show_tracks=True):
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.imshow(vid[t], origin='lower')
    if show_tracks:
        plt.plot(tracks[t, :, 0], tracks[t, :, 1], 'rx')
    plt.xlim(-10, 265)
    plt.ylim(-10, 265)

    plt.subplot(122)
    plt.imshow(vid[t], origin='lower')
    plt.imshow(labels[t, ..., 1], origin='lower')
    plt.show()

# Predict z-coordinates for a test frame
predicted_z_coords = model_Z_Position.predict(patches)
num_results_to_display = 100

# Assuming 'history' is the variable holding the return value from the model's fit function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training loss on the first subplot
ax1.plot(epochs, loss, 'bo-', label='Training loss')
ax1.set_title('Training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot validation loss on the second subplot
ax2.plot(epochs, val_loss, 'r-', label='Validation loss')
ax2.set_title('Validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()

for actual, predicted in zip(z_coords[:num_results_to_display], predicted_z_coords[:num_results_to_display]):
    print(f"Actual Z: {actual}, Predicted Z: {predicted[0]}")
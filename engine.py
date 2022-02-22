"""
Script for ENGINE: Enhancing Neuroimaging and Genetic Information by Neural Embedding framework
Written in Tensorflow 2.1.0
"""

# Import APIs
import tensorflow as tf
import numpy as np

class engine(tf.keras.Model):
    tf.keras.backend.set_floatx('float32')
    """ENGINE framework"""

    def __init__(self, N_o):
        super(engine, self).__init__()
        self.N_o = N_o # the number of classification outputs

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(2098,)),  # F_SNP = 2098
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=100, activation=None, kernel_regularizer='L1L2'),  # 2 * dim(z_SNP) = 100
            ]
        )

        # Decoder network, P
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)), # dim(z_SNP) = 50
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=2098, activation=None, kernel_regularizer='L1L2'), # F_SNP = 2098
            ]
        )

        """Attentive Vector Generation Module"""
        # Generator network, G
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(54,)), # dim(z) = dim(z_SNP) + dim(c) = 54
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=186, activation='sigmoid', kernel_regularizer='L1L2'), # 2 * F_MRI = 186
            ]
        )

        # Discriminator network, D
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(93,)), # F_MRI = 93
                tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='L1L2'), # real or fake
            ]
        )

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(93,)), # dim(Concat(a, x_MRI)) = 93
                tf.keras.layers.Dense(units=25, activation='elu', kernel_regularizer='L1L2'),
            ]
        )

        self.diagnostician_clf = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25, )), # dim(f) = 25
                tf.keras.layers.Dense(units=self.N_o, activation=None, kernel_regularizer='L1L2'),  # |N_o|
            ]
        )

        self.diagnostician_reg = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25,)),  # dim(f) = 25
                tf.keras.layers.Dense(units=1, activation=None, kernel_regularizer='L1L2'),  # 1
            ]
        )

    @tf.function
    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, 50))
        return self.decode(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean, logvar = tf.split(self.encoder(x_SNP), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Construct latent distribution
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.math.exp(logvar * .5) + mean

    # Reconstruct the input SNP
    def decode(self, z_SNP, apply_sigmoid=False):
        logits = self.decoder(z_SNP)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits

    # Attentive vector and fake neuroimaging generation
    def generate(self, z_SNP, c_demo):
        z = tf.concat([c_demo, z_SNP], axis=-1)
        x_MRI_fake, a = tf.split(self.generator(z), num_or_size_splits=2, axis=1)
        return x_MRI_fake, a

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        return self.discriminator(x_MRI_real_or_fake)

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_MRI, a, apply_logistic_activation=False):
        feature = self.diagnostician_share(tf.multiply(x_MRI, a)) # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = tf.math.softmax(logit_clf)
            s_hat = tf.math.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

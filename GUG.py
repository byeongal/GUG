import keras
from keras import layers

class GUG :
    def __init__(self):
        # Unpacker
        self.packed_file = keras.Input(shape=(2000000,))
        self.x = layers.Reshape((2000000, 1))(self.packed_file)
        self.x = layers.Conv1D(256, 4, padding='same')(self.x)
        self.x = layers.LeakyReLU()(self.x)
        self.x = layers.Conv1D(256, 4, padding='same')(self.x)
        self.x = layers.LeakyReLU()(self.x)
        self.x = layers.Conv1D(1, 4, padding='same')(self.x)
        self.unpacker = keras.models.Model(self.packed_file, self.x)

        # Discriminator
        self.discriminator_input = layers.Input(shape=(2000000, 1))
        self.x = layers.Conv1D(128, 4, padding='same')(self.discriminator_input)
        self.x = layers.LeakyReLU()(self.x)
        self.x = layers.Conv1D(128, 4, padding='same')(self.x)
        self.x = layers.LeakyReLU()(self.x)
        self.x = layers.Flatten()(self.x)
        self.x = layers.Dense(1, activation='sigmoid')(self.x)
        self.discriminator = keras.models.Model(self.discriminator_input, self.x)
        self.discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss='binary_crossentropy')

        self.gug_input = keras.Input(shape=(2000000, ))
        self.gug_output = self.discriminator(self.unpacker(self.gug_input))
        self.gug = keras.models.Model(self.gug_input, self.gug_output)
        self.gug_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        self.gug.compile(optimizer=self.gug_optimizer, loss='binary_crossentropy')

    def train(self, packed_file_path_list, origin_file_path_list):
        pass

    def unpack(self, packed_file_path) :
        pass

import numpy as np

import keras
from keras import layers
from keras.preprocessing import sequence

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

        # GUG( General Unpacker using GAN )
        self.gug_input = keras.Input(shape=(2000000, ))
        self.gug_output = self.discriminator(self.unpacker(self.gug_input))
        self.gug = keras.models.Model(self.gug_input, self.gug_output)
        self.gug_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
        self.gug.compile(optimizer=self.gug_optimizer, loss='binary_crossentropy')

    def load_file(self, file_path_list) :
        ret = []
        for file_path in file_path_list :
            with open(file_path, 'rb') as f :
                ret.append(f.read())
        return sequence.pad_sequences(ret, maxlen=2000000)

    def train(self, packed_file_path_list, origin_file_path_list):
        start = 0
        batch_size = 16
        step = 0
        while True :
            stop = start + batch_size
            origin_files = self.load_file(origin_file_path_list[start:stop])
            packed_files = self.load_file(packed_file_path_list[start:stop])

            unpakced_files = self.unpacker.predict(packed_files)

            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            labels += 0.05 * np.random.random(labels.shape)

            combined_images = np.concatenate([origin_files, unpakced_files])
            d_loss = self.discriminator.train_on_batch(combined_images, labels)

            misleading_targets = np.zeros((batch_size, 1))
            a_loss = self.gug.train_on_batch(packed_files, misleading_targets)

            start += batch_size
            if start > len(packed_file_path_list) - batch_size :
                start = 0

            if step % 1000 == 0 :
                print('스텝 %s에서 판별자 손실: %s' % (step, d_loss))
                print('스텝 %s에서 적대적 손실: %s' % (step, a_loss))
                self.gug.save('gug.h5')
                self.discriminator.save('discriminator.h5')
                self.unpacker.save('unpacker.h5')

    def unpack(self, packed_file_path) :
        pass

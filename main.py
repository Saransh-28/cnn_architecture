from tensorflow.keras import *
from tensorflow.keras.models import *

# alexnet from scratch using tensorflow
def AlexNet():
  inp = layers.Input((224, 224, 3))
  x = layers.Conv2D(96, 11, 4, activation='relu')(inp)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(3, 2)(x)
  x = layers.Conv2D(256, 5, 1, activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.MaxPooling2D(3, 2)(x)
  x = layers.Conv2D(384, 3, 1, activation='relu')(x)
  x = layers.Conv2D(384, 3, 1, activation='relu')(x)
  x = layers.Conv2D(256, 3, 1, activation='relu')(x)
  x = layers.MaxPooling2D(3, 2)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(4096, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=inp, outputs=x)

  return model


# vggnet from scratch using tensorflow
def VGGnet():
    def vgg_block(x, num_layers=2, out_channels=64):
        for _ in range(num_layers):
            x = Conv2D(out_channels, 3, 1, padding='same')(x)
            x = ReLU()(x)
        x = MaxPooling2D(2, 2)(x)
        return x
    inp = Input((224, 224, 3))
    block_1 = vgg_block(inp, num_layers=2, out_channels=64)
    block_2 = vgg_block(block_1, num_layers=2, out_channels=128)
    block_3 = vgg_block(block_2, num_layers=3, out_channels=256)
    block_4 = vgg_block(block_3, num_layers=3, out_channels=512)
    block_5 = vgg_block(block_4, num_layers=3, out_channels=512)
    flat = Flatten()(block_5)
    fc_1 = Dense(4096, activation='relu')(flat)
    fc_2 = Dense(4096, activation='relu')(fc_1)
    output = Dense(1, activation='sigmoid')(fc_2)
    
    model = Model(inputs=inp, outputs=output)
    
    return model

# inception net from scratch using tensorflow
def inceptionnet():
    def inception_block(x, base_channels=32):
        a = Conv2D(base_channels*2, 1, 1, activation='relu')(x)
        b_1 = Conv2D(base_channels*2, 1, 1, activation='relu')(x)
        b_2 = Conv2D(base_channels*4, 3, 1, padding='same', activation='relu')(b_1)
        c_1 = Conv2D(base_channels, 1, 1, activation='relu')(x)
        c_2 = Conv2D(base_channels, 5, 1, padding='same', activation='relu')(c_1)
        d_1 = MaxPooling2D(3, 1, padding='same')(x)
        d_2 = Conv2D(base_channels, 1, 1, activation='relu')(d_1)

        return Concatenate(axis=-1)([a, b_2, c_2, d_2])

    inp = Input((224, 224, 3))
    block_1 = inception_block(inp)
    block_2 = inception_block(block, base_channels=16)

    gap_2d = GlobalAveragePooling2D()(block)
    output = Dense(1, activation='sigmoid')(gap_2d)

    model = Model(inputs=inp, outputs=output)
    
    return model



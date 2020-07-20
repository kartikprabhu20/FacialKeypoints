from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow as tf

class FCNmodel(tf.keras.Model):

  def __init__(self):
    super(FCNmodel, self).__init__()
    
    # Initialising the CNN
    self.model = Sequential()

    # 1 - Convolution
    self.model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))
    
    # 2nd Convolution layer
    self.model.add(Conv2D(128,(5,5), padding='same'))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    # 3rd Convolution layer
    self.model.add(Conv2D(512,(3,3), padding='same'))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    # 4th Convolution layer
    self.model.add(Conv2D(512,(3,3), padding='same'))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))

    # Flattening
    self.model.add(Flatten())

    # Fully connected layer 1st layer
    self.model.add(Dense(256))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    self.model.add(Dense(512))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(Dropout(0.25))

    self.model.add(Dense(7, activation='softmax'))

  def call(self, input):
       return self.model(input)
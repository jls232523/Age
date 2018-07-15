from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class cnnModel:
    @staticmethod
    def build(imgHeight,imgWidth,channel,ageGrps):
        model = Sequential()
        imgShape = (imgHeight,imgWidth,channel)
        chanDim = -1;
        # first convolution layer to a rectifier into a pool
        # 32 filters 3x3
        model.add(Conv2D(50, (4, 4), padding="same",input_shape=imgShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.25))
        #randomly disconnects nodes while training in order to cause redundancy
        model.add(Conv2D(100, (4, 4), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(100, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #64 filter size w/ 2x2
        """model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #128 filter 3x3"""
        #  fully connected layer
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(ageGrps))
        model.add(Activation("softmax"))#classifys image based on greatest probability

        # return the constructed network architecture
        return model

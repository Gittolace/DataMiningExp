from keras.layers import Dropout, Dense

from keras.models import Sequential


class myModel:
    def build(input_shape, classes):
        model = Sequential()
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation="softmax"))

        return model

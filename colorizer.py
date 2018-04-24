import easyargs
import numpy as np
import pandas as pd
from IPython.display import Image, display
from keras.layers import Conv2D, InputLayer, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from matplotlib.pyplot import imshow, imsave
from skimage.color import lab2rgb, rgb2lab

@easyargs
def main(src_image_path, dest_image_path):
    src_image_xy = turn_image_into_input_and_output(src_image_path)
    dest_image_xy = turn_image_into_input_and_output(dest_image_path)
    model = create_and_train_model(*src_image_xy)
    colorize_image(dest_image_xy[0], model)

def turn_image_into_input_and_output(image_path, target_size=(200,200)):
    image = img_to_array(load_img(image_path, target_size=target_size)) / 255
    lab_image = rgb2lab(image)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]

    # The input will be the black and white layer
    X = lab_image_norm[:,:,0]
    
    # The outpts will be the ab channels
    Y = lab_image_norm[:,:,1:]

    # The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
    # (samples, rows, cols, channels), so we need to do some reshaping
    # https://keras.io/layers/convolutional/
    X = X.reshape(1, X.shape[0], X.shape[1], 1)
    Y = Y.reshape(1, Y.shape[0], Y.shape[1], 2)

    return X,Y

def create_and_train_model(X, Y, epochs=100):
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3,3), activation='sigmoid', padding='same'))

    # Finish model
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x=X, y=Y, batch_size=1, epochs=epochs, verbose=1)

    return model

def colorize_image(X, model, name='result.jpg'):
    output = model.predict(X)
    cur = np.zeros((200, 200, 3))
    cur[:,:,0] = X[0][:,:,0]
    cur[:,:,1:] = output[0]


    #imshow(rgb_image.astype('float32'))

    #cur = (cur * [100, 255, 255]) - [0 ,128, 128]
    #output

    #rgb_image *= [100, 255, 255]
    cur = (cur * [100, 255, 255]) - [0, 128, 128]
    rgb_image = lab2rgb(cur)
    imsave(name, rgb_image)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # Params will be handled by Easyargs, if the script is called by the command line
    main()
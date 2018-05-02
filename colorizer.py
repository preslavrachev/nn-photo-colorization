import easyargs
import numpy as np
import os
import pandas as pd
import random
import re
from IPython.display import Image, display
from keras.layers import Conv2D, InputLayer, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from matplotlib.pyplot import imshow, imsave
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2lab

IMAGE_SIZE = (512, 512)

@easyargs
def main(src_images_dir_path, target_image_path, output_image_path='result.jpg'):
    paths = find_all_images_suitable_for_input(
        src_images_dir_path, exclude_paths=[target_image_path, output_image_path])
    images = [load_and_resize_image(path, target_size=IMAGE_SIZE) for path in paths]
    crops = [generate_randomly_cropped_image(image, size=IMAGE_SIZE)
             for k in range(1, 10)
             for image in images]
    src_images_xy = load_all_inputs(images)
    model = create_and_train_model(*src_images_xy)

    # prepare the black and white image
    dest_image = load_and_resize_image(target_image_path, target_size=IMAGE_SIZE)
    dest_image_xy = turn_image_into_input_and_output(dest_image)

    # colorrize the black and white image using the model
    colorize_image(dest_image_xy[0], model)


def find_all_images_suitable_for_input(input_dir_path, exclude_paths=[]):
    return [input_dir_path + fn
            for fn in os.listdir(input_dir_path)
            if re.match(r'.*\.jpg', fn) and fn not in exclude_paths]


def load_all_inputs(images) -> (np.array, np.array):
    '''
    Loads all input images from the given directoy, and turns them into
    respective model inputs and outputs
    '''
    all_x_y = [turn_image_into_input_and_output(image) for image in images]

    all_x = np.array(
        list(map(lambda xy: np.array(xy[0]).reshape(*xy[0].shape), all_x_y)))
    all_y = np.array(
        list(map(lambda xy: np.array(xy[1]).reshape(*xy[1].shape), all_x_y)))
    return all_x, all_y


def turn_image_into_input_and_output(image):
    '''
    Loads, resizes, and converts input images into inputs and outputs for the NN model
    '''
    lab_image = rgb2lab(image)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]

    # The input will be the black and white layer
    X = lab_image_norm[:, :, 0]

    # The outpts will be the ab channels
    Y = lab_image_norm[:, :, 1:]

    # The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
    # (samples, rows, cols, channels), so we need to do some reshaping
    # https://keras.io/layers/convolutional/
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 2)

    return X, Y


def load_and_resize_image(image_path, target_size=None) -> Image:
    '''
    # Returns
    # The loaded and resized PIL image
    '''
    img = Image.open(image_path)

    if target_size is None:
        return img
        else:
        return ImageOps.fit(img, target_size, Image.ANTIALIAS)


def generate_randomly_cropped_image(original_image, size):
    original_image_w, original_image_h = original_image.size
    crop_w = size[0]
    a = random.randint(0, original_image_w - crop_w)
    crop_h = size[1]
    b = random.randint(0, original_image_h - crop_h)
    return original_image.crop(box=(a, b, a + crop_w, b + crop_h))


    input_width = X.shape[1]
    input_height = X.shape[2]
    input_channels = 1 # this is only the L channel (B&W)

    model = Sequential()
    model.add(InputLayer(input_shape=(input_width, input_height, input_channels)))
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
    model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))

    # Finish model
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x=X, y=Y, batch_size=4, epochs=epochs, verbose=1)

    return model


def colorize_image(X, model, name='result.jpg'):
    X = X.reshape(1, *X.shape)
    output = model.predict(X)
    cur = np.zeros((output.shape[1], output.shape[2], 3))
    cur[:, :, 0] = X[0][:, :, 0]
    cur[:, :, 1:] = output[0]

    cur = (cur * [100, 255, 255]) - [0, 128, 128]
    rgb_image = lab2rgb(cur)
    imsave(name, rgb_image)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # Params will be handled by Easyargs, if the script is called by the command line
    main()

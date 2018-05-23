import easyargs
import numpy as np
import os
import random
import re
import sklearn.neighbors as nn
from IPython.display import Image
from keras.callbacks import TensorBoard
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, InputLayer, Lambda, UpSampling2D
from keras.models import Sequential
from keras.utils import to_categorical
from matplotlib.pyplot import imsave
from PIL import ImageOps
from skimage.color import lab2rgb, rgb2lab

IMAGE_SIZE = (128, 128)
PTS_IN_HULL = np.load('../input/pts_in_hull.npy')
PTS_IN_HULL_ONE_HOT_ENCODED = to_categorical(
    np.arange(0, PTS_IN_HULL.shape[0]))


@easyargs
def main(src_images_dir_path, target_image_path, output_image_path='result.jpg'):
    paths = find_all_images_suitable_for_input(
        src_images_dir_path, exclude_paths=[target_image_path, output_image_path])
    images = [load_and_resize_image(
        path, target_size=IMAGE_SIZE) for path in paths]
    src_images_xy = load_all_inputs(images)
    model = create_and_train_model(*src_images_xy)

    # prepare the black and white image
    dest_image = load_and_resize_image(
        target_image_path, target_size=IMAGE_SIZE)
    dest_image_xy = turn_image_into_input_and_output(dest_image)

    # colorrize the black and white image using the model
    colorize_image(dest_image_xy[0], model)


def find_all_images_suitable_for_input(input_dir_path, exclude_paths=[]):
    input_dir_path = input_dir_path[:-
                                    1] if input_dir_path[-1] == '/' else input_dir_path
    exclude_fns = [os.path.basename(exclude_path) for exclude_path in exclude_paths if os.path.dirname(
        exclude_path) in input_dir_path]
    return [input_dir_path + '/' + fn
            for fn in os.listdir(input_dir_path)
            if re.match(r'.*\.[jpg|jpeg]', fn) and fn not in exclude_fns]


def load_all_inputs(images) -> (np.array, np.array):
    '''
    Loads all input images from the given directoy, and turns them into
    respective model inputs and outputs
    '''
    all_x_y = [turn_image_into_input_and_output(image) for image in images]

    all_x = np.array(
        list(map(lambda xy: np.array(xy[0]), all_x_y)))
    all_y = np.array(
        list(map(lambda xy: np.array(xy[1]), all_x_y)))
    return all_x, all_y


def quantize(inputs, to_points, to_points_one_hot_encoded):
    neighbors = nn.NearestNeighbors(
        n_neighbors=10, algorithm='auto').fit(to_points)
    dists, indices = neighbors.kneighbors(inputs)

    end_points = np.zeros((inputs.shape[0], to_points.shape[0]))
    sigma = 5.0
    wts = np.exp(-dists**2/(2*sigma**2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    end_points[np.arange(0, inputs.shape[0], dtype='int')
               [:, np.newaxis], indices] = wts
    return end_points


def turn_image_into_input_and_output(image):
    '''
    Loads, resizes, and converts input images into inputs and outputs for the NN model
    '''
    lab_image = rgb2lab(image)
    lab_image_norm = (lab_image + [0, 0, 0]) / [100, 1, 1]

    # The input will be the black and white layer
    X = lab_image_norm[:, :, 0]

    # The outpts will be the ab channels
    Y = lab_image_norm[:, :, 1:]

    # The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
    # (samples, rows, cols, channels), so we need to do some reshaping
    # https://keras.io/layers/convolutional/
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 2)
    Y = quantize(Y.reshape(-1, 2), to_points=PTS_IN_HULL,
                 to_points_one_hot_encoded=PTS_IN_HULL_ONE_HOT_ENCODED)
    Y = Y.reshape(X.shape[0], X.shape[1], PTS_IN_HULL.shape[0])

    return [X, Y]


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


def load_q_kernels(shape):
    print(shape)
    samples = np.random.choice(
        ((PTS_IN_HULL + [128, 128]) / [255, 255])[0], shape[3] * shape[2])
    # return((PTS_IN_HULL + [128, 128]) / [255, 255]).reshape(shape)
    return samples.reshape(shape)


def create_and_train_model(X, Y, epochs=3000):
    input_width = X.shape[1]
    input_height = X.shape[2]
    input_channels = 1  # this is only the L channel (B&W)

    mul = 12

    model = Sequential()
    model.add(InputLayer(input_shape=(input_width, input_height, input_channels)))
    model.add(Dropout(0.1))
    model.add(Conv2D(8 * mul, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8 * mul, (3, 3), activation='relu',
                     padding='same', strides=2))
    model.add(Conv2D(16 * mul, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16 * mul, (3, 3), activation='relu',
                     padding='same', strides=2))
    model.add(Conv2D(32 * mul, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32 * mul, (3, 3), activation='relu',
                     padding='same', strides=2))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32 * mul, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16 * mul, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(313, (1, 1)))
    model.add(Lambda(lambda x: x / 0.38))
    model.add(Activation('softmax'))

    # Finish model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    tensorboard = TensorBoard(log_dir="./tensorboard")

    model.fit(x=X, y=Y, batch_size=6, epochs=epochs,
              callbacks=[tensorboard], verbose=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('weights_dump.h5')

    return model


def xy_gen(x, y, datagen):
    for batch in datagen.flow(x=x, y=y, batch_size=4):
        yield(batch[0], batch[1])


def colorize_image(X, model, name='result.jpg') -> Image:
    # model.load_weights('weights_dump_ultra.h5')
    X = X.reshape(1, *X.shape)
    output = model.predict(X).reshape(-1, 313)
    output = np.dot(output, PTS_IN_HULL).reshape(1, X.shape[1], X.shape[2], 2)
    cur = np.zeros((output.shape[1], output.shape[2], 3))
    cur[:, :, 0] = X[0][:, :, 0]
    cur[:, :, 1:] = output[0]

    cur = (cur * [100, 1, 1])
    print(output.min())
    rgb_image = (255*np.clip(lab2rgb(cur), 0, 1)).astype('uint8')
    colored_image = imsave(name, rgb_image)

    return colored_image


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    # Params will be handled by Easyargs, if the script is called by the command line
    main()

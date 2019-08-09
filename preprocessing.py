
import os
import csv
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt

def read_csv(filename, delimiter=','):

    csv_data = dict()
    keys = []

    with open(filename) as f:

        csv_reader = csv.reader(f, delimiter=delimiter)

        for i, row in enumerate(csv_reader):

            if i == 0:
                keys = row
                for key in keys:
                    csv_data[key] = []
            else:
                for key, data in zip(keys, row):
                    csv_data[key].append(data)

        print('Processed {} lines.'.format(i))

    return csv_data, i

def get_random_x(filenames, labels, directory, amount=1, extension=".png"):

    images = {}

    unique_labels = np.unique(labels)

    while len(images) < len(unique_labels):
        random_file = random.choice(filenames)
        random_file_index = filenames.index(random_file)
        random_file_label = labels[random_file_index]

        if random_file_label not in images: 
            f = os.path.join(directory, random_file_label, random_file + extension)
            image = cv.imread(f)
            images[str(random_file_label)] = image

    return images

def imshow_1x2(image_1, image_2=None, show=False):

    if image_2 is None:
        image_2 = image_1

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image_1)
    ax2.imshow(image_2)

    if show:
        plt.show()

def crop(image):

    if len(image.shape) == 3:
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img = image

    img = auto_canny(image)
    x, y = np.where(img > 0)

    try:
        up = x[np.argmin(x)]
        down = x[np.argmax(x)]
        left = y[np.argmin(y)]
        right = y[np.argmax(y)]
    except:
        # imshow_1x2(image, show=True)
        print("up: {}, down: {}, left: {}, right: {}".format(up, down, left, right))
        return image
        
    return image[up:down+1, left:right+1]

def auto_canny(image, sigma=0.33):

    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)

    return edged

def preprocess_image(image, grayscale=False, width=224, height=224, output_channels=None, auto_crop=True):

    if(isinstance(image, str)):
        img = cv.imread(image)
    else:
        img = image

    if auto_crop:
        img = crop(img)

    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)

    if grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0,0) , (width+height)/20), -4, 128)
        
        if output_channels is None:
            img = np.expand_dims(img, axis=2)
        else:
            img = np.stack([img] * output_channels, axis=2)
    else:
        img = cv.addWeighted(img, 4, cv.GaussianBlur(img, (0,0) , (width+height)/20), -4, 128)

    return img


if __name__ == "__main__":

    random.seed(50)

    train_dir = "/media/hdd/data/train_images"
    test_dir = "/media/hdd/data/test_images"
    train_csv = "/media/hdd/data/train.csv"
    test_csv = "/media/hdd/data/test.csv"
    densenet_weights = '/media/hdd/data/densenet/DenseNet-BC-121-32-no-top.h5'
    
    data, N = read_csv(train_csv)
    images = get_random_x(data["id_code"], data["diagnosis"], train_dir)

    for label in images:
        image = images[label]
        p_image = preprocess_image(image, grayscale=False, output_channels=3)
        imshow_1x2(image, p_image)

    plt.show()




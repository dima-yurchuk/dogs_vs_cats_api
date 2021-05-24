import os, cv2
import numpy as np

# %matplotlib inline
import pickle
# TRAIN_FOLDER = './train/'
TEST_FOLDER = 'static/images/'

# train_images = [TRAIN_FOLDER + i for i in os.listdir("./train/")]  # use this for full dataset
# test_images = [TEST_FOLDER + i for i in os.listdir("./test/")]

ROWS = 64
COLS = 64
CHANNELS = 3


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE
    #     print(img.shape)
    #     grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    """
    Returns:
        X(n_x, m)
        y(1, m) -- 1: dog, 0: cat
    """
    m = len(images)
    n_x = ROWS * COLS * CHANNELS

    X = np.ndarray((n_x, m), dtype=np.uint8)
    y = np.zeros((1, m))
    print("X shape is {}".format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[:, i] = np.squeeze(image.reshape((n_x, 1)))
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
        else:  # if neither dog nor cat exist, return the image index (this is the case for test data)
            y[0, i] = 3
        if i % 1000 == 0: print('Processed {} of {}'.format(i, m))

    return X, y


# X_train, Y_train = prep_data(train_images)
# X_test, test_idx = prep_data(test_images)

# print("Train shape: {}".format(X_train.shape))
# print("Test shape: {}".format(X_test.shape))

classes = {0: 'cat',
           1: 'dog'}



# def show_image_prediction(X, idx, model):
#     image = X[idx].reshape(1, -1)
#     image_class = classes[model.predict(image).item()]
#     image = image.reshape((ROWS, COLS, CHANNELS))
#     plt.figure(figsize=(4, 2))
#     plt.imshow(image)
#     plt.title("Test {}: This is a {}".format(idx, image_class))
#     plt.show()


# X, Y = X_train.T, Y_train.T.ravel()
# x_train =  X[0:int(len(X) * 0.9)]
# y_train =  Y[0:int(len(Y) * 0.9)]
# x_test =  X[int(len(X) * 0.9):]
# y_test =  Y[int(len(Y) * 0.9):]
# print(Y[0:30])
#
# model = LogisticRegression(max_iter=30)
# model.fit(x_train, y_train)
#
filename = 'static/model.sav'
# pickle.dump(model, open(filename, 'wb'))

# load the model from disk
model = pickle.load(open(filename, 'rb'))


def prediction():
    test_images = [TEST_FOLDER + i for i in os.listdir("static/images/")]
    X_test, test_idx = prep_data(test_images)
    X_test = X_test.T
    image = X_test[0].reshape(1, -1)
    image_class = classes[model.predict(image).item()]
    return image_class
    # print(image_class)
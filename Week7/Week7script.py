# %%
%matplotlib qt
from matplotlib import pyplot as plt
from numpy.random import rand
import skimage as sm
import skimage.morphology as morph
from skimage.io import imread, imsave
from skimage.util import img_as_float, random_noise
from skimage.transform import rotate, resize
from skimage.filters import gaussian as ski_gaussian
from skimage.feature import canny, corner_harris, corner_peaks
from pylab import ginput
from scipy.signal import convolve, convolve2d, correlate2d, fftconvolve
from scipy.signal import  gaussian as scipy_gaussian
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates, gaussian_filter

import numpy as np
import os
import timeit
import tensorflow as tf


os.chdir("../Mats")

# %%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %% [markdown]
# # 1

# %%
def straight_line_Hough(image):
    diag_len = np.sqrt(image.shape[0]**2+image.shape[1]**2)
    Hough_transform = np.zeros((2*np.ceil(diag_len).astype(int)+1, 180))
    radians = np.linspace(-90, 89, 180).astype(int)
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[i, j] != 0:
                for theta in radians:
                    p = np.ceil(diag_len + i*np.cos(np.pi*theta/180) + j*np.sin(np.pi*theta/180)).astype(int)
                    Hough_transform[p, 90+theta] += 1
    return Hough_transform

# %% [markdown]
# 1.2

# %%
A = imread("cross.png")
radians_array = np.linspace(-90, 89, 180).astype(int)*np.pi/180
diag_len = np.ceil(np.sqrt(A.shape[0]**2+A.shape[1]**2)).astype(int)
p_array = np.linspace(-diag_len, diag_len, 2*diag_len+1)
own_hough = straight_line_Hough(A)
ski_hough, angle, d = hough_line(A)

ticks_angle = np.linspace(0, own_hough.shape[1], 5)
label_angle = np.linspace(-90, 90, 5).astype(int)
plt.subplot(1, 3, 1)
plt.imshow(own_hough, cmap="gray")
plt.xticks(ticks_angle, label_angle)
plt.xlabel("\u03B8")
plt.ylabel("p")
plt.title("own function")
plt.colorbar(shrink=0.5)
plt.subplot(1, 3, 2)
plt.imshow(ski_hough, cmap="gray")
plt.xticks(ticks_angle, label_angle)
plt.xlabel("\u03B8")
plt.ylabel("p")
plt.title("skimage function")
plt.colorbar(shrink=0.5)
plt.subplot(1, 3, 3)
plt.imshow(np.abs(ski_hough-own_hough), cmap="gray")
plt.xticks(ticks_angle, label_angle)
plt.xlabel("\u03B8")
plt.ylabel("p")
plt.title("Difference")
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.plot()

# %%
own_coord = np.unravel_index(np.argsort(own_hough.ravel()), own_hough.shape)
own_angles = radians_array[own_coord[1][-2:]]
own_p = p_array[own_coord[0][-2:]]
ski_coord = np.unravel_index(np.argsort(ski_hough.ravel()), ski_hough.shape)
ski_angles = radians_array[ski_coord[1][-2:]]
ski_p = p_array[ski_coord[0][-2:]]
x_axis = np.linspace(0, A.shape[1]-1, 1000)
own_y_1 = (own_p[0]-np.cos(own_angles[0])*x_axis)/np.sin(own_angles[0])
own_y_2 = (own_p[1]-np.cos(own_angles[1])*x_axis)/np.sin(own_angles[1])
ski_y_1 = (ski_p[0]-np.cos(ski_angles[0])*x_axis)/np.sin(ski_angles[0])
ski_y_2 = (ski_p[1]-np.cos(ski_angles[1])*x_axis)/np.sin(ski_angles[1])
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="gray")
plt.plot(x_axis, own_y_1, linewidth=0.7, color="r")
plt.plot(x_axis, own_y_2, linewidth=0.7, color="r")
plt.title("own function")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(A, cmap="gray")
plt.plot(x_axis, ski_y_1, linewidth=0.7, color="r")
plt.plot(x_axis, ski_y_2, linewidth=0.7, color="r")
plt.title("skimage function")
plt.axis("off")
plt.tight_layout() 
plt.show()

# %%
print("Angles for own function:", own_angles)
print("p for own function:", own_p)
print("Angles for ski function:", ski_angles)
print("p for ski function:", ski_p)

# %% [markdown]
# 1.3

# %%
A = imread("coins.png")
A_edge = canny(A, sigma=3)

A_circles = hough_circle(A_edge, radius=1, full_output=True)
plt.imshow(A_circles[0,:,:], cmap="gray")
plt.title("Hough circle segmentation of coins.png")
plt.axis("off")
plt.show()

# %% [markdown]
# # 2

# %% [markdown]
# 2.1

# %%


# %% [markdown]
# 2.2

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
K.clear_session()
## Configure the network
# batch_size to train
batch_size = 20 * 256
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 400

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential([
    InputLayer(input_shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Flatten(),
    Dense(units=4000, activation='relu'),
    Dense(units=nb_classes, activation='softmax'),
])
    
optimizer = Adam(lr=1e-4, epsilon=1e-08)

model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.load_weights('keras.h5') 

test = np.load(r'C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\test.npz')
x_test= test['x_test']
y_test= test['y_test']
x_test = np.reshape(x_test, (x_test.shape[0], 29, 29, 1))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)


# %%
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

# %% [markdown]
# 2.3

# %%
def image_patches(image):
    padded_image = np.pad(image, ((14, 14), (14, 14)), 'constant')

    patches = sm.util.shape.view_as_windows(padded_image, (29, 29), step=1)

    return patches.reshape(-1, 29, 29, 1)

# %%
image = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\image\1025_3_image.png")

patches = image_patches(image)
patches = patches / 255
segmented_img = []
pr = model.predict(patches)
dim = int(pr.shape[0]**0.5)
for i, prediction in enumerate(pr):
    segmented_img.append(np.argmax(prediction))
segmented_img = np.array(segmented_img).reshape((dim, dim))

# %%
plt.subplot(1,2,1)
plt.imshow(A,cmap="gray"),plt.axis("off"),plt.title("Original input")
plt.subplot(1,2,2)
plt.imshow(segmented_img,cmap="gray"),plt.axis("off"),plt.title("Predicted segmentation")
plt.tight_layout()
plt.show()

# %% [markdown]
# 2.4

# %%
def dice_coef(pred_seg, true_seg):
    pred_seg = np.asarray(pred_seg).astype(bool)
    true_seg = np.asarray(true_seg).astype(bool)
    intersection = np.logical_and(pred_seg, true_seg)
    dice = (2. * intersection.sum()) / (pred_seg.sum() + true_seg.sum())
    return dice

segImage = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\seg\1025_3_seg.png")

dice_coef(segmented_img,segImage)
plt.subplot(1,2,1)
plt.imshow(segImage,cmap="gray"),plt.axis("off"),plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(segmented_img,cmap="gray"),plt.axis("off"),plt.title("Predicted segmentation")
plt.show()

# %%
def dice_coef(pred_seg, true_seg):
    pred_seg = np.asarray(pred_seg).astype(bool)
    true_seg = np.asarray(true_seg).astype(bool)
    intersection = np.logical_and(pred_seg, true_seg)
    dice = (2. * intersection.sum()) / (pred_seg.sum() + true_seg.sum())
    return dice

# %%
def dice_coef(pred_seg, true_seg):
    pred_seg = np.asarray(pred_seg).astype(bool)
    true_seg = np.asarray(true_seg).astype(bool)
    intersection = np.logical_and(pred_seg, true_seg)
    dice = (2. * intersection.sum()) / (pred_seg.sum() + true_seg.sum())
    return dice

diceVals = []
for i in range(135):
    testimage = np.zeros(segImage.shape)
    testimage2 = np.zeros(segmented_img.shape)
    mask = segImage == i
    testimage[mask] = segImage[mask]
    mask = segmented_img == i
    testimage2[mask] = segmented_img[mask]
    dice = dice_coef(testimage2,testimage)

    diceVals.append(dice_coef(testimage2,testimage))


diceVals = [x for x in diceVals if str(x) != 'nan']
print(diceVals)


# %% [markdown]
# 2.5

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
K.clear_session()
## Configure the network
# batch_size to train
batch_size = 20 * 256
# number of output classes
nb_classes = 80
# number of epochs to train
nb_epoch = 400

# number of convolutional filters to use
nb_filters = 10
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

model = Sequential([
    InputLayer(input_shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='relu'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.2),
    Flatten(),
    Dense(units=4000, activation='relu'),
    Dense(units=nb_classes, activation='softmax'),
])
    
optimizer = Adam(lr=1e-4, epsilon=1e-08)

model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])



#model.load_weights('keras.h5') 



# %%
A = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\image\1003_3_image.png")
segImage = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\seg\1003_3_seg.png")

patches = image_patches(A)
segmented_img = []
pr = model.predict(patches)
dim = int(pr.shape[0]**0.5)
for i, prediction in enumerate(pr):
    segmented_img.append(np.argmax(prediction))
    
segmented_img = np.array(segmented_img).reshape((dim, dim))
plt.subplot(1,2,1)
plt.imshow(A,cmap="gray"),plt.axis("off"),plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(segmented_img,cmap="gray"),plt.axis("off"),plt.title("Predicted segmentation")
plt.show()
dice_coef(segmented_img,segImage)

# %% [markdown]
# 0.625256054946379

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
K.clear_session()
## Configure the network
# batch_size to train
batch_size = 10 * 256
# number of output classes
nb_classes = 135
# number of epochs to train
nb_epoch = 400

# number of convolutional filters to use
nb_filters = 20
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

model = Sequential([
    InputLayer(input_shape=(29, 29, 1)),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='tanh'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.5),
    Conv2D(filters=nb_filters, kernel_size=nb_conv, activation='tanh'),
    MaxPool2D(pool_size=(nb_pool, nb_pool)),
    Dropout(0.25),
    Flatten(),
    Dense(units=4000, activation='tanh'),
    Dense(units=nb_classes, activation='softmax'),
])
    
optimizer = Adam(lr=1e-4, epsilon=1e-08)

model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.load_weights('keras.h5') 

test = np.load(r'C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\test.npz')
x_test= test['x_test']
y_test= test['y_test']
x_test = np.reshape(x_test, (x_test.shape[0], 29, 29, 1))


# %%
A = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\image\1025_3_image.png")
segImage = imread(r"C:\Users\Jakob\Desktop\Uni\Kandidat\ISP\SIP\Mats\test_images\seg\1025_3_seg.png")

patches = image_patches(A)
patches = patches /255
segmented_img = []
pr = model.predict(patches)
dim = int(pr.shape[0]**0.5)
for i, prediction in enumerate(pr):
    segmented_img.append(np.argmax(prediction))
    
segmented_img = np.array(segmented_img).reshape((dim, dim))
plt.subplot(1,2,1)
plt.imshow(A,cmap="gray"),plt.axis("off"),plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(segmented_img,cmap="gray"),plt.axis("off"),plt.title("Predicted segmentation")
plt.show()
dice_coef(segmented_img,segImage)

# %%
diceVals = []
for i in range(135):
    testimage = np.zeros(segImage.shape)
    testimage2 = np.zeros(segmented_img.shape)
    mask = segImage == i
    testimage[mask] = segImage[mask]
    mask = segmented_img == i
    testimage2[mask] = segmented_img[mask]
    dice = dice_coef(testimage2,testimage)

    diceVals.append(dice_coef(testimage2,testimage))


diceVals = [x for x in diceVals if str(x) != 'nan']
print(np.mean(diceVals))

# %% [markdown]
# 0.6898007576414794 And
# 0.1954659989409428

# %%




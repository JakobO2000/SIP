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


os.chdir("../Mats")

# %%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %% [markdown]
# # 1

# %% [markdown]
# 1.1

# %%
A = imread("hand.tiff")
plt.imshow(A, cmap="gray"), plt.title("hand.tiff"), plt.axis("off")
plt.show()

# %%
B = canny(A, 1, 25, 100)
C = canny(A, 3, 25, 100)
D = canny(A, 1, 75, 100)
E = canny(A, 1, 25, 150)
plt.subplot(2, 2, 1)
plt.imshow(B, cmap="gray"), plt.title("\u03C3=1 low_t=25 high_t=100", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(C, cmap="gray"), plt.title("\u03C3=3 low_t=25 high_t=100", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(D, cmap="gray"), plt.title("\u03C3=1 low_t=75 high_t=100", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(E, cmap="gray"), plt.title("\u03C3=1 low_t=25 high_t=150", fontsize=7), plt.axis("off")
plt.show()

# %% [markdown]
# 1.2

# %%
A = imread("modelhouses.png")

k_A = corner_harris(A, "k", 0.02, sigma=1)
k_B = corner_harris(A, "k", 0.08, sigma=1)
k_C = corner_harris(A, "k", 0.02, sigma=3)
k_D = corner_harris(A, "k", 0.08, sigma=3)
print(k_A)
plt.subplot(2, 2, 1)
plt.imshow(k_A, vmin=0, vmax=1, cmap="gray"), plt.title("k=0.02 \u03C3=1", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(k_B, vmin=0, vmax=1,cmap="gray"), plt.title("k=0.08 \u03C3=1", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(k_C, vmin=0, vmax=1,cmap="gray"), plt.title("k=0.02 \u03C3=3", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(k_D, vmin=0, vmax=1,cmap="gray"), plt.title("k=0.08 \u03C3=3", fontsize=7), plt.axis("off")
plt.suptitle("Method k")
plt.show()

# %%
A = imread("modelhouses.png")
eps_A = corner_harris(A, "eps", eps=1, sigma=1)
eps_B = corner_harris(A, "eps", eps=1e-6, sigma=1)
eps_C = corner_harris(A, "eps", eps=1, sigma=3)
eps_D = corner_harris(A, "eps", eps=1e-6, sigma=3)
plt.subplot(2, 2, 1)
plt.imshow(eps_A, vmin=0, vmax=1, cmap="gray"), plt.title("eps=1 \u03C3=1", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(eps_B, vmin=0, vmax=1,cmap="gray"), plt.title("eps=1e-6 \u03C3=1", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(eps_C, vmin=0, vmax=1,cmap="gray"), plt.title("eps=1 \u03C3=3", fontsize=7), plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(eps_D, vmin=0, vmax=1,cmap="gray"), plt.title("eps=1e-6 \u03C3=3", fontsize=7), plt.axis("off")
plt.suptitle("Method eps")
plt.show()

# %% [markdown]
# 1.3

# %%
def find_corners(image, N_corners, method="k", k=0.05, eps=1e-06, sigma=1):
    image_corner = corner_harris(image, method, k, eps, sigma)
    return corner_peaks(image_corner, num_peaks=N_corners)

A = imread("modelhouses.png")
corners = find_corners(A, 250) 

plt.imshow(A, cmap="gray")
plt.scatter(corners[:,1], corners[:,0], s=0.5), plt.axis("off")
plt.title("250 strongest corner points")
plt.show()


# %% [markdown]
# # 2

# %% [markdown]
# 2.1

# %%
def angle_between_points(p1,p2):
    delta_x = p2[1] - p1[1]
    delta_y = p2[0] - p1[0]
    return np.arctan2(delta_y, delta_x)


A  = imread("textlabel_gray_small.png")

corners = find_corners(A,50,k=0.05,sigma=7)
top_left = corners[np.argmin(np.sum(corners, axis=1))]
top_right = corners[np.argmin(np.diff(corners, axis=1))]
bottom_right = corners[np.argmax(np.sum(corners, axis=1))]
bottom_left = corners[np.argmax(np.diff(corners, axis=1))]
indices = []
for corner in [top_left, top_right, bottom_right, bottom_left]:
    index = np.where((corners == corner).all(axis=1))[0][0]
    indices.append(index)
corners = np.delete(corners, indices, axis=0)
top_left = corners[np.argmin(np.sum(corners, axis=1))]
bottom_left = corners[np.argmin(np.diff(corners, axis=1))]
bottom_right = corners[np.argmax(np.sum(corners, axis=1))]
top_right = corners[np.argmax(np.diff(corners, axis=1))]
Ang1 = angle_between_points(bottom_left,top_left)
Ang2 = angle_between_points(bottom_right,top_right)
AngAvg = (Ang1 + Ang2) / 2
AngDegr = np.rad2deg(AngAvg)
rotated_image = rotate(A,(AngDegr),resize=True)


plt.subplot(1,2,1)
plt.imshow(A,cmap="gray")
plt.scatter(corners[:,1], corners[:,0], s=1), plt.axis("off")
plt.scatter(top_left[1],top_left[0], s=10,color="red"), plt.axis("off")
plt.scatter(top_right[1],top_right[0], s=10,color="green"), plt.axis("off")
plt.scatter(bottom_right[1],bottom_right[0], s=10,color="green"), plt.axis("off")
plt.scatter(bottom_left[1],bottom_left[0], s=10,color="red"), plt.axis("off")


plt.subplot(1,2,2)
plt.imshow(rotated_image,cmap="gray"),plt.axis("off")
plt.tight_layout()
plt.show()



# %%
AngDegr

# %%
Ang1 = angle_between_points(bottom_left,top_left)
Ang2 = angle_between_points(bottom_right,top_right)
AngAvg = (Ang1 + Ang2) / 2
AngDegr = np.rad2deg(AngAvg)
rotated_image = rotate(A,(AngDegr),resize=True)
plt.imshow(rotated_image,cmap="gray")
plt.show()

# %% [markdown]
# # 3

# %% [markdown]
# 3.1

# %%
def gaussian(x, y, sigma):
    return (1/(2*np.pi*sigma**2))*np.e**(-(x**2 + y**2)/(2*sigma**2))

def generate_image(size, func, func_args):
    img = np.zeros((size, size))
    
    half_size = int((size / 2))
    rng = range(-half_size, half_size+1)
    for i in rng:
        for j in rng:
            img[i,j] = func(i, j, func_args)
    
    img = np.roll(img, shift = (half_size, half_size), axis = (0, 1))
    
    return img

size = 25
func = gaussian
sigma = 1
tau = 2

blob_s = generate_image(size, func, sigma)
blob_t = generate_image(size, func, tau)
blob_conv = convolve2d(blob_s, blob_t, mode = 'same', boundary = 'wrap')
blob_comparison = generate_image(size, func, np.sqrt(sigma**2 + tau**2))

vmin, vmax = 0, np.max(blob_s)

plot_images = [blob_s, blob_t, blob_conv, blob_comparison, np.abs(np.subtract(blob_comparison, blob_conv))]
plot_labels = ['$G_\\sigma(x,y,\\sigma)$', '$G_\\tau(x,y,\\tau)$', '$G_\\sigma * G_\\tau$', '$G(x,y,\\sqrt{\\sigma^2 + \\tau^2})$', '$G - (G_\\sigma * G_\\tau)$']
fig, axs = plt.subplots(1,5)
for i, image in enumerate(plot_images):
    im = axs[i].imshow(image, 'gray')
    axs[i].set_axis_off()
    axs[i].set_title(plot_labels[i], fontsize = 8)
    cbar = plt.colorbar(im, ax = axs[i], shrink = 1, orientation = 'horizontal', pad = 0.02)
    cbar.ax.tick_params(labelsize = 5)
          
plt.savefig('Blobs.png', dpi = 200)
plt.show()

# %%
plot_images = [blob_s, blob_t, blob_conv, blob_comparison, np.abs(np.subtract(blob_comparison, blob_conv))]
plot_labels = ['$G_\\sigma(x,y,\\sigma)$', '$G_\\tau(x,y,\\tau)$', '$G_\\sigma * G_\\tau$', '$G(x,y,\\sqrt{\\sigma^2 + \\tau^2})$', '$G - (G_\\sigma * G_\\tau)$']
fig, axs = plt.subplots(1,5)
for i, image in enumerate(plot_images):
    im = axs[i].imshow(image, 'gray', vmin = 0, vmax = vmax)
    axs[i].set_axis_off()
    axs[i].set_title(plot_labels[i], fontsize = 8)
    cbar = plt.colorbar(im, ax = axs[i], shrink = 1, orientation = 'horizontal', pad = 0.02)
    cbar.ax.tick_params(labelsize = 5)
          
plt.savefig('BlobsEqualScale.png', dpi = 200)
plt.show()

# %% [markdown]
# 3.2

# %%
#Done in maple, written in on overleaf

# %% [markdown]
# 3.3

# %% [markdown]
# i

# %%
#Done in maple doc

# %% [markdown]
# ii

# %%
#Done in maple doc

# %% [markdown]
# iii

# %%
tau = np.linspace(-10, 10, 1000)
H = -tau**2/(np.pi*(4+tau**2))

plt.plot(tau, H)
plt.xlabel("\u03C4")
plt.ylabel("H(0, 0, \u03C4)")
plt.title("H(0, 0, \u03C4) plotted for various \u03C4")
plt.tight_layout()
plt.show()

# %% [markdown]
# 3.4

# %%
def blob_detection(img,scales):
    L = np.zeros((img.shape[0], img.shape[1], len(scales)))
    for i, s in enumerate(scales):
        L[:,:,i] = sm.filters.laplace(sm.filters.gaussian(img,sigma=s))
    scaleSpace = np.zeros((img.shape[0], img.shape[1], len(scales)))
    for i, s in enumerate(scales):
        scaleSpace[:,:,i] = s**2 *(L[:,:,i])
    maxCoords = sm.feature.peak_local_max(scaleSpace, num_peaks=150)
    minCoords = sm.feature.peak_local_max(scaleSpace.max()-scaleSpace,num_peaks=150)
    return maxCoords, minCoords



# %%

A = imread("sunflower.tiff",as_gray=True)

scales = np.linspace(0,30,100)

maxCoords,minCoords = blob_detection(A,scales)
svalsmin = []
svalsmax = []
fig, ax = plt.subplots()
ax.imshow(A,cmap="gray")
for coord in maxCoords:
    x, y, z = coord
    r = scales[z]
    svalsmin.append(r)
    circle = plt.Circle((y, x), r, color='red', fill=False,linewidth=0.5)
    dot = plt.plot(y, x, 'ro', markersize=0.5)
    ax.add_artist(circle)


for coord in minCoords:
    x, y, z = coord
    r = scales[z]
    svalsmax.append(r)
    circle = plt.Circle((y, x), r, color='blue', fill=False,linewidth=0.5)
    dot = plt.plot(y, x, 'bo', markersize=0.5)
    ax.add_artist(circle)

plt.axis("off")

plt.show()

# %%
min(svalsmin),max(svalsmin),min(svalsmax),max(svalsmax)

# %% [markdown]
# 



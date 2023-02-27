# %%
%matplotlib qt
from matplotlib import pyplot as plt
from numpy.random import rand
import skimage as sm
from skimage.io import imread, imsave
from skimage.util import img_as_float, random_noise
from skimage.transform import rotate, resize
from skimage.filters import gaussian as ski_gaussian
from pylab import ginput
from scipy.signal import convolve, convolve2d, correlate2d, fftconvolve
from scipy.signal import  gaussian as scipy_gaussian
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import os
import timeit


os.chdir("../Mats")

# %%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %% [markdown]
# # 2

# %% [markdown]
# 2.1

# %%

def apply_filter(image, filter):
    cop_image = np.copy(image)
    out_image = np.zeros(image.shape)
    row = (filter.shape[0]-1)//2
    column = (filter.shape[1]-1)//2
    for i in range(cop_image.shape[0]-filter.shape[0]):
        for j in range(cop_image.shape[1]-filter.shape[1]):
            out_image[i+row,j+column] = np.sum(cop_image[i:i+filter.shape[0],j:j+filter.shape[1]]*filter)
    return out_image


def fft_convolve(image, filter):
    cop_image = np.copy(image)
    prod = fft2(image, image.shape) * fft2(filter, image.shape)
    ifftprod = ifft2(prod).real
    convImg = np.roll(ifftprod, (-((filter.shape[0] - 1)//2),
                                 -((filter.shape[1] - 1)//2)), axis=(0, 1))
    return convImg


def mean_kernel(N):
    return np.ones((N, N))/N**2

A = imread("cameraman.tif")
kernels = [mean_kernel(3), mean_kernel(7)]
print(A.shape)

img_reg_3 = apply_filter(A, kernels[0])
img_fft_3 = fft_convolve(A, kernels[0]) 
img_reg_7 = apply_filter(A, kernels[1])
img_fft_7 = fft_convolve(A, kernels[1])
diff_3 = np.abs(img_fft_3-img_reg_3)
diff_7 = np.abs(img_fft_7-img_reg_7)

reg_kernel_time = []
fft_kernel_time = []

for i in range(2, 16):
    A = imread("cameraman.tif")
    kernel = mean_kernel(i)
    reg_kernel_time.append(timeit.timeit(lambda: apply_filter(A, kernel), number=100))
    fft_kernel_time.append(timeit.timeit(lambda: fft_convolve(A, kernel), number=100))
    print(i)


# %%
np.save("reg_kernel_time.npy", np.array(reg_kernel_time))
np.save("fft_kernel_time.npy", np.array(fft_kernel_time))
reg_kernel_time = np.load("reg_kernel_time.npy")
fft_kernel_time = np.load("fft_kernel_time.npy")

# %%
fig, ax = plt.subplots()
x_axis = np.arange(2, 16)
ax.plot(x_axis, reg_kernel_time, label="reg")
ax.plot(x_axis, fft_kernel_time, label="fft")
ax.set_xlabel("N"), ax.set_ylabel("Execution time (s)"), ax.set_title("Execution time for 100 executions for a filter with kernel NxN")
plt.tight_layout()
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots(3, 2, figsize=(2.1, 3))
ax[0, 0].imshow(img_reg_3, vmin=0, vmax=255, cmap="gray"), ax[0,0].set_title("Nested for, N=3", fontsize=6), ax[0, 0].axis("off")
ax[0, 1].imshow(img_reg_7, vmin=0, vmax=255, cmap="gray"), ax[0,1].set_title("Nested for, N=7", fontsize=6), ax[0, 1].axis("off")
ax[1, 0].imshow(img_fft_3, vmin=0, vmax=255, cmap="gray"), ax[1,0].set_title("fft, N=3", fontsize=6), ax[1, 0].axis("off")
ax[1, 1].imshow(img_fft_7, vmin=0, vmax=255, cmap="gray"), ax[1,1].set_title("fft, N=7", fontsize=6), ax[1, 1].axis("off")
ax[2, 0].imshow(diff_3, vmin=0, vmax=255, cmap="gray"), ax[2,0].set_title("Difference, N=3", fontsize=6), ax[2, 0].axis("off")
ax[2, 1].imshow(diff_7, vmin=0, vmax=255, cmap="gray"), ax[2,1].set_title("Difference, N=7", fontsize=6), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %%
A = imread("cameraman.tif")
kernel = mean_kernel(5)
img = [resize(A, (50, 50)), resize(A, (200, 200))]
img_reg_small = apply_filter(img[0], kernel)
img_fft_small = fft_convolve(img[0], kernel)
img_reg_big = apply_filter(img[1], kernel)
img_fft_big = fft_convolve(img[1], kernel)
diff_3 = np.abs(img_fft_small-img_reg_small)
diff_7 = np.abs(img_reg_big-img_fft_big)

reg_img_time = []
fft_img_time = []

for i in range(20, 256, 10):
    img = resize(A, (i, i))
    reg_img_time.append(timeit.timeit(lambda: apply_filter(img, kernel), number=100))
    fft_img_time.append(timeit.timeit(lambda: fft_convolve(img, kernel), number=100))
    print(i)

# %%
np.save("reg_img_time.npy", np.array(reg_img_time))
np.save("fft_img_time.npy", np.array(fft_img_time))
reg_img_time = np.load("reg_img_time.npy")
fft_img_time = np.load("fft_img_time.npy")

# %%
fig, ax = plt.subplots()
x_axis = np.arange(20, 256, 10)
ax.plot(x_axis, reg_img_time, label="reg")
ax.plot(x_axis, fft_img_time, label="fft")
ax.set_xlabel("N"), ax.set_ylabel("Execution time (s)"), ax.set_title("Execution time for 100 executions for an image with NxN pixels")
plt.tight_layout()
plt.legend()
plt.show()


# %%
fig, ax = plt.subplots(3, 2, figsize=(2.1, 3))
ax[0, 0].imshow(img_reg_small,  cmap="gray"), ax[0,0].set_title("Nested for, 50x50 img", fontsize=5), ax[0, 0].axis("off")
ax[0, 1].imshow(img_reg_big,  cmap="gray"), ax[0,1].set_title("Nested for, 150x150 img", fontsize=5), ax[0, 1].axis("off")
ax[1, 0].imshow(img_fft_small, cmap="gray"), ax[1,0].set_title("fft, 50x50 img", fontsize=5), ax[1, 0].axis("off")
ax[1, 1].imshow(img_fft_big, cmap="gray"), ax[1,1].set_title("fft, 150x150 img", fontsize=5), ax[1, 1].axis("off")
ax[2, 0].imshow(diff_3, cmap="gray"), ax[2,0].set_title("Difference, 50x50 img", fontsize=5), ax[2, 0].axis("off")
ax[2, 1].imshow(diff_7, cmap="gray"), ax[2,1].set_title("Difference, 150x150 img", fontsize=5), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# 2.2

# %%
def waveAdd(image,a,v,w):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    cos_wave = a * np.cos(v * x + w * y)
    img = image + cos_wave
    return img



def powerSpec(image):
    ft = fft2(np.copy(image))
    ftshift = fftshift(ft)
    ps = ftshift**2
    return np.abs(ps)

# %%
A = imread("cameraman.tif")
a, v, w = 50, 0.5, 0.5
B = waveAdd(A,a,v,w)

fftA = fft2(A)
fftA = fftshift(fftA)
fftB = fft2(B)
fftB = fftshift(fftB)
ps_A = powerSpec(A)
ps_B = powerSpec(B)

plt.subplot(2,2,1)
plt.imshow(A,cmap="gray"),plt.axis("off")
plt.title("Original")
plt.subplot(2,2,2)
plt.imshow(B,cmap="gray"),plt.axis("off")
plt.title("Original + noise")
plt.subplot(2,2,3)
plt.imshow(np.log10(1+ps_A),cmap="gray"),plt.xticks([], []),plt.yticks([], [])
plt.title("PS of original")
plt.subplot(2,2,4)
plt.imshow(np.log10(1+ps_B),cmap="gray"),plt.xticks([], []),plt.yticks([], [])
plt.title("PS + noise")
plt.show()

# %%
#107, 107
#148, 148

# %%
def filterFunc(fft,pixel1,pixel2):
    N = fftB.shape[0]
    x,y = np.meshgrid(np.arange(N), np.arange(N))
    a1, a2 = 0.005, 0.005
    F1 = 1 - np.exp(-a1*(x-pixel1[0])**2-a2*(y-pixel1[1])**2)
    F2 = 1 - np.exp(-a1*(x-pixel2[0])**2-a2*(y-pixel2[1])**2)
    Z = F1*F2
    imgFs = fft*Z
    imgF = ifftshift(imgFs)
    imgF = ifft2(imgF)
    return imgF, Z

# %%
pixel1 = [107,107]
pixel2 = [148,148]
imgF, Z = filterFunc(fftB,pixel1,pixel2)

plt.subplot(2,2,1)
plt.imshow(B,cmap="gray"),plt.axis("off")
plt.title("Original + noise")
plt.subplot(2,2,2)
plt.imshow(np.real(imgF),cmap="gray"),plt.axis("off")
plt.title("Filtered image")
plt.subplot(2,2,3)
plt.imshow(np.log10(1+np.abs(fftB)),cmap="gray"),plt.xticks([], []),plt.yticks([], [])
plt.title("FFT + noise")
plt.subplot(2,2,4)
plt.imshow(Z,cmap="gray"),plt.xticks([], []),plt.yticks([], [])
plt.title("Filter")
plt.show()

# %%




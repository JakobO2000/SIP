# %%
%matplotlib qt
from matplotlib import pyplot as plt
from numpy.random import rand
import skimage as sm
from skimage.io import imread, imsave
from skimage.util import img_as_float, random_noise
from skimage.transform import rotate
from skimage.filters import gaussian as ski_gaussian
from pylab import ginput
from scipy.signal import convolve, convolve2d, correlate2d
from scipy.signal import  gaussian as scipy_gaussian
from scipy.fft import fft2, fftshift
import numpy as np
import os
import timeit

os.chdir("../Mats")

# %%
def plotGray(image): #Plots an image in gray scale
    plt.axis("off")
    plt.imshow(image,cmap="gray")

def plotRGB(image): #Plots an image in RGB color
    plt.axis("off")
    plt.imshow(image,cmap="jet")

# %% [markdown]
# # 1

# %% [markdown]
# 1.1

# %%
def gammaFuncGray(image,gamma):
    return 255*(image/255)**gamma

# %%
def gammaFuncRGB(image,gamma):
    img = np.copy(image)
    for i in range(image.shape[2]):
        img[:,:,i] = 255*(img[:,:,i]/255)**gamma
    return img

# %%
def gammaFunc(image,gamma,HSV=False):
    img = np.copy(image)
    
    try:
        for i in range(image.shape[2]):
            img[:,:,i] = 255*(img[:,:,i]/255)**gamma
        return img
    except:
        return 255*(image/255)**gamma

# %%
A = imread("cameraman.tif")

B=gammaFunc(A,0.5)
C=gammaFunc(A,2)


plt.subplot(1,3,1)
plotGray(A)
plt.title("Original Image")
plt.subplot(1,3,2)
plotGray(B)
plt.title("Gamma = 0.5 Image")
plt.subplot(1,3,3)
plotGray(C)
plt.title("Gamma = 2 Image")
plt.tight_layout()
plt.show()

# %% [markdown]
# 1.2

# %%
A = imread("autumn.tif")
B=gammaFunc(A,0.5)
C=gammaFunc(A,2)

plt.subplot(1,3,1)
plotGray(A)
plt.title("Original Image")
plt.subplot(1,3,2)
plotGray(B)
plt.title("Gamma = 0.5 Image")
plt.subplot(1,3,3)
plotGray(C)
plt.title("Gamma = 2 Image")
plt.tight_layout()
plt.show()

# %% [markdown]
# 1.3

# %%
def RBGtoHSV(image):
    img = np.copy(img_as_float(image))
    HSV_M = np.zeros(img.shape)
    H = np.zeros((img.shape[0],img.shape[1]))
    S = np.zeros((img.shape[0],img.shape[1]))
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    theta = np.arccos(((2*R-B-G)/2) / np.sqrt(R**2+G**2+B**2 - R*B - R*G - G*B + 1e-12))
    mask = img[:,:,2]>img[:,:,1]
    H[mask] = 2*np.pi - theta[mask]
    H[~mask] = theta[~mask]
        
    V =  np.amax(img,axis=2)
    mask = V > 0
    S[mask] = (V[mask]-np.amin(img,axis=2)[mask])/V[mask]
    S[~mask] = 0
    HSV_M[:,:,0] = H
    HSV_M[:,:,1] = S
    HSV_M[:,:,2] = V
    return HSV_M


# %%
def gammaHSV(image,gamma):
    img = sm.color.rgb2hsv(image)
    img[:,:,2] = (img[:,:,2])**gamma
    return sm.color.hsv2rgb(img)

# %%
A = imread("autumn.tif")
B=gammaHSV(A,0.5)
C=gammaHSV(A,2)

plt.subplot(1,3,1)
plotGray(A)
plt.title("Original Image")
plt.subplot(1,3,2)
plotGray(B)
plt.title("Gamma = 0.5 Image")
plt.subplot(1,3,3)
plotGray(C)
plt.title("Gamma = 2 Image")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 2

# %% [markdown]
# 2.1

# %%
def median_filter_optimized(image, N):
    pad_size = (N) // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='symmetric')
    output = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel = padded_image[i:i+N, j:j+N].ravel()
            kernel.sort()
            median_index = (N*N - 1) // 2
            output[i, j] = kernel[median_index]

    return output

def mean_filter_optimized(image, N):
    cop_image = np.copy(image)
    kernel = np.ones((N, N)) / N ** 2
    return convolve2d(cop_image, kernel, mode='same')


# %%
def median_filter(image, N):
    mir_image = np.pad(image, np.int32(np.floor(N/2)+1), mode="symmetric")
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = np.median(mir_image[i:i+N,j:j+N].flatten())
    return image

def mean_filter(image, N):
    cop_image = np.copy(image)
    kernel = np.ones(N)/N
    mir_image = np.pad(cop_image, np.int32(np.floor(N/2)+1), mode="symmetric")
    for i in range(cop_image.shape[0]):
        for j in range(cop_image.shape[1]):
            cop_image [i,j] = np.sum(mir_image[i:i+N,j:j+N]@kernel@kernel.T)
    return cop_image
    
A = imread("eight.tif")
executions = 100
N_max = 25
A_snp = random_noise(A, "s&p")
A_gaus = random_noise(A)
plt.subplot(1, 3, 1), plt.imshow(A, vmin=0, vmax=255, cmap="gray"), plt.title("Original")
plt.subplot(1, 3, 2), plt.imshow(A_snp, cmap="gray"), plt.title("s&p")
plt.subplot(1, 3, 3), plt.imshow(A_gaus, cmap="gray"), plt.title("Gaussian")
plt.show()

# %%
N = 3

snp_mean_3 = mean_filter(np.copy(A_snp), N)
snp_median_3 = median_filter_optimized(np.copy(A_snp), N)
gaus_mean_3 = mean_filter(np.copy(A_gaus), N)
gaus_median_3 = median_filter_optimized(np.copy(A_gaus), N)
N = 7
snp_mean_7 = mean_filter(np.copy(A_snp), N)
snp_median_7 = median_filter_optimized(np.copy(A_snp), N)
gaus_mean_7 = mean_filter(np.copy(A_gaus), N)
gaus_median_7 = median_filter_optimized(np.copy(A_gaus), N)

fig, ax = plt.subplots(3, 2, dpi=300, figsize=(4, 12))
fig.suptitle("Mean filter", fontsize=16)
ax[0, 0].imshow(A_snp, vmin=0, vmax=1,cmap="gray"), ax[0, 0].set_title("S&p noise"), ax[0, 0].axis("off")
ax[0, 1].imshow(A_gaus, vmin=0, vmax=1,cmap="gray"), ax[0, 1].set_title("Gaussian noise"), ax[0, 1].axis("off")
ax[1, 0].imshow(snp_mean_3, vmin=0, vmax=1,cmap="gray"), ax[1, 0].set_title("N = 3"), ax[1, 0].axis("off")
ax[1, 1].imshow(gaus_mean_3, vmin=0, vmax=1,cmap="gray"), ax[1, 1].set_title("N = 3"), ax[1, 1].axis("off")
ax[2, 0].imshow(snp_mean_7, vmin=0, vmax=1,cmap="gray"), ax[2, 0].set_title("N = 7"), ax[2, 0].axis("off"), ax[2, 0].set_xlabel("N = 5")
ax[2, 1].imshow(gaus_mean_7, vmin=0, vmax=1,cmap="gray"), ax[2, 1].set_title("N = 7"), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(3, 2, dpi=300, figsize=(4, 12))
fig.suptitle("Median filter", fontsize=16)
ax[0, 0].imshow(A_snp, vmin=0, vmax=1,cmap="gray"), ax[0, 0].set_title("S&p noise"), ax[0, 0].axis("off")
ax[0, 1].imshow(A_gaus, vmin=0, vmax=1,cmap="gray"), ax[0, 1].set_title("Gaussian noise"), ax[0, 1].axis("off")
ax[1, 0].imshow(snp_median_3, vmin=0, vmax=1,cmap="gray"), ax[1, 0].set_title("N = 3"), ax[1, 0].axis("off")
ax[1, 1].imshow(gaus_median_3, vmin=0, vmax=1,cmap="gray"), ax[1, 1].set_title("N = 3"), ax[1, 1].axis("off")
ax[2, 0].imshow(snp_median_7, vmin=0, vmax=1,cmap="gray"), ax[2, 0].set_title("N = 7"), ax[2, 0].axis("off")
ax[2, 1].imshow(gaus_median_7, vmin=0, vmax=1,cmap="gray"), ax[2, 1].set_title("N = 7"), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %%
def mean_100(image, N):
        for j in range(100):
            mean_filter(image, N)
        return

def median_100(image, N):
        for j in range(100):
           median_filter_optimized(image, N)
        return
  
mean_time = []
median_time = []
for i in range(1, N_max+1):
    print(i)
    start_time = timeit.default_timer()
    mean_100(np.copy(A), i)
    mean_time.append(timeit.default_timer()-start_time)
    start_time = timeit.default_timer()
    median_100(np.copy(A), i)
    median_time.append(timeit.default_timer()-start_time)



# %%
# np.save("mean_time_data_2.npy", np.array(mean_time))
# np.save("median_time_data_2.npy", np.array(median_time))
mean_data = np.load("mean_time_data_2.npy")
median_data = np.load("median_time_data_2.npy")

# %%
x_axis = np.arange(1, 26)
fig, ax = plt.subplots()
ax.plot(x_axis, mean_data, label="Mean time")
ax.plot(x_axis, median_data, label="Median time")
ax.set_ylabel("time (s)")
ax.set_xlabel("N")
ax.set_title("Execution time for 100 executions of filters with NxN kernel")
plt.legend()
plt.show()

# %% [markdown]
# 2.2

# %%
def gaussian_filter(image, N, sigma):
    cop_image = np.copy(image)
    gaussian = scipy_gaussian(N, sigma)
    kernel = np.outer(gaussian, gaussian)
    mir_image = np.pad(cop_image, np.int32(np.floor(N/2)+1), mode="symmetric")
    for i in range(cop_image.shape[0]):
        for j in range(cop_image.shape[1]):
            cop_image [i,j] = np.sum(mir_image[i:i+N,j:j+N]@gaussian@gaussian.T)
    return cop_image

def gaussian_filter_optimized(image, N, sigma):
    cop_image = np.copy(image)
    gaussian = scipy_gaussian(N, sigma)
    kernel = np.outer(gaussian, gaussian)
    return convolve2d(cop_image, kernel)
sigma = 2
snp_gaus_3 = gaussian_filter_optimized(A_snp, 2, sigma)
gaus_gaus_3 = gaussian_filter_optimized(A_gaus, 2, sigma)
snp_gaus_5 = gaussian_filter_optimized(A_snp, 5, sigma)
gaus_gaus_5 = gaussian_filter_optimized(A_gaus, 5, sigma)



# %%
fig, ax = plt.subplots(3, 2, dpi=300, figsize=(4, 12))
fig.suptitle("Gaussian filter with \u03C3 = 5", fontsize=16)
ax[0, 0].imshow(A_snp, vmin=0, vmax=1,cmap="gray"), ax[0, 0].set_title("S&p noise"), ax[0, 0].axis("off")
ax[0, 1].imshow(A_gaus, vmin=0, vmax=1,cmap="gray"), ax[0, 1].set_title("Gaussian noise"), ax[0, 1].axis("off")
ax[1, 0].imshow(snp_gaus_3, cmap="gray"), ax[1, 0].set_title("N = 3"), ax[1, 0].axis("off")
ax[1, 1].imshow(gaus_gaus_3, cmap="gray"), ax[1, 1].set_title("N = 3"), ax[1, 1].axis("off")
ax[2, 0].imshow(snp_gaus_5, cmap="gray"), ax[2, 0].set_title("N = 5"), ax[2, 0].axis("off")
ax[2, 1].imshow(gaus_gaus_5, cmap="gray"), ax[2, 1].set_title("N = 5"), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# 2.3

# %%
sigma = 1
snp_gaus_s3 = gaussian_filter_optimized(A_snp, 3*sigma, sigma)
gaus_gaus_s3 = gaussian_filter_optimized(A_gaus, 3*sigma, sigma)
sigma = 3
snp_gaus_s7 = gaussian_filter_optimized(A_snp, 3*sigma, sigma)
gaus_gaus_s7 = gaussian_filter_optimized(A_gaus, 3*sigma, sigma)

fig, ax = plt.subplots(3, 2, dpi=300, figsize=(4, 12))
fig.suptitle("Gaussian filter with N=3\u03C3", fontsize=16)
ax[0, 0].imshow(A_snp, vmin=0, vmax=1,cmap="gray"), ax[0, 0].set_title("S&p noise"), ax[0, 0].axis("off")
ax[0, 1].imshow(A_gaus, vmin=0, vmax=1,cmap="gray"), ax[0, 1].set_title("Gaussian noise"), ax[0, 1].axis("off")
ax[1, 0].imshow(snp_gaus_s3, cmap="gray"), ax[1, 0].set_title("\u03C3 = 1"), ax[1, 0].axis("off")
ax[1, 1].imshow(gaus_gaus_s3, cmap="gray"), ax[1, 1].set_title("\u03C3 = 1"), ax[1, 1].axis("off")
ax[2, 0].imshow(snp_gaus_s7, cmap="gray"), ax[2, 0].set_title("\u03C3 = 3"), ax[2, 0].axis("off")
ax[2, 1].imshow(gaus_gaus_s7, cmap="gray"), ax[2, 1].set_title("\u03C3 = 3"), ax[2, 1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 3

# %%
# Exercise 3 has been completed in the overleaf, see https://www.overleaf.com/6782353343wcfmggcfzyyd

# %% [markdown]
# # 4

# %% [markdown]
# 4.1

# %%
def powerSpec(image):
    ft = fft2(np.copy(image))
    ftshift = fftshift(ft)
    ps = ftshift**2
    return ps



# %%
A = imread("trui.png")

ps_A = powerSpec(A)
plt.subplot(1, 2, 1), plt.imshow(A, cmap="gray"), plt.title("Original"),plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(np.log10(1+np.abs(ps_A)), cmap="gray"), plt.title("Power Spectrum"),plt.axis("off")
plt.tight_layout()
plt.show()

# %%
B = rotate(A,-30, resize=True)
ps_B = powerSpec(B)
plt.subplot(2, 2, 1), plt.imshow(B, cmap="gray"), plt.title("Original rotated 30 degrees"),plt.axis("off")
plt.subplot(2, 2, 2), plt.imshow(np.log10(1+np.abs(ps_B)), cmap="gray"), plt.title("30 degrees Power Spectrum"),plt.axis("off")
B = rotate(A,-45, resize=True)
ps_B = powerSpec(B)
plt.subplot(2, 2, 3), plt.imshow(B, cmap="gray"), plt.title("Original rotated 45 degrees"),plt.axis("off")
plt.subplot(2, 2, 4), plt.imshow(np.log10(1+np.abs(ps_B)), cmap="gray"), plt.title("45 degrees Power Spectrum"),plt.axis("off")

plt.tight_layout()
plt.show()

# %%




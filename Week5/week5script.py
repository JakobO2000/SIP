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
from pylab import ginput
from scipy.signal import convolve, convolve2d, correlate2d, fftconvolve
from scipy.signal import  gaussian as scipy_gaussian
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

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

# %% [markdown]
# 1.1.1

# %%
A = imread("cells_binary_inv.png")

disk = morph.disk(1)
openA = morph.binary_opening(A,disk)
closeA = morph.binary_closing(A,disk)

plt.subplot(2,3,1)
plt.imshow(A,cmap="gray"),plt.axis("off"),plt.title("Original")
plt.subplot(2,3,2)
plt.imshow(openA,cmap="gray"),plt.axis("off"),plt.title("Opening")
plt.subplot(2,3,3)
plt.imshow(closeA,cmap="gray"),plt.axis("off"),plt.title("Closing")
plt.subplot(2,3,4)
plt.imshow(A,cmap="gray"),plt.axis("off"),plt.title("Original"),plt.xlim(100,200),plt.ylim(150,250)
plt.subplot(2,3,5)
plt.imshow(openA,cmap="gray"),plt.axis("off"),plt.title("Opening"),plt.xlim(100,200),plt.ylim(150,250)
plt.subplot(2,3,6)
plt.imshow(closeA,cmap="gray"),plt.axis("off"),plt.title("Closing"),plt.xlim(100,200),plt.ylim(150,250)

# %% [markdown]
# 1.1.2

# %%


# %% [markdown]
# 1.1.3

# %%
A = imread("cells_binary_inv.png")
disk = morph.disk(1)
openA = morph.binary_opening(A,disk)
closedA = morph.binary_closing(A,disk)
connectedopen = sm.measure.label(openA, background=255,connectivity=2)
connectedclosed = sm.measure.label(closedA, background=255,connectivity=2)
connectedA = sm.measure.label(A, background=255,connectivity=2)

plt.subplot(1,2,1)
plt.imshow(connectedopen,cmap="gray"),plt.axis("off"),plt.title("Open connected ({})".format(connectedopen.max()))
plt.subplot(1,2,2)
plt.imshow(connectedclosed,cmap="gray"),plt.axis("off"),plt.title("Closed connected ({})".format(connectedclosed.max()))

print(connectedopen.max(),connectedclosed.max(),connectedA.max())

# %% [markdown]
# 1.2

# %%
def check_positions(m1, m2, N):
    mask = np.array(m1) == N
    result = np.any(np.array(m2)[mask] != 0)
    return result
A = imread("money_bin.jpg")
mask = A > 230
A[mask],A[~mask] = 0, 1
disk = morph.disk(5)
Atest = morph.binary_closing(A)
Atest = morph.binary_opening(Atest)
connectedA = sm.measure.label(Atest, background=255, connectivity=2)
coinVals,cIndex,Loops,coinSum,uq =[1,2,5,0.5,20],0,100,0,np.unique(connectedA)[1:]
for i in range(Loops):
    Atest,remCount = morph.binary_erosion(Atest, disk),0
    for j in uq:
        if not check_positions(connectedA, Atest, j):
            uq = uq[uq != j]
            remCount +=1
    if remCount != 0:
        coinSum += remCount*coinVals[cIndex]
        cIndex +=1
print("Coinsum = ", coinSum)

# %%
plt.imshow(connectedA,cmap="jet")
plt.show()

# %% [markdown]
# # 2

# %% [markdown]
# 2.1

# %%


# %%
def gaussian_kernel(size=5, sig=1.):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# %%
def LSI(img,kernel,noise):
    return convolve2d(img,kernel,mode="same",boundary="wrap") + noise


A = imread("monster.jpg",as_gray=True)
A = imread("trui.png",as_gray=True)

noise = [0,50]
kernels = [gaussian_kernel(9,sig=1), gaussian_kernel(18,sig=5)]
for i, noise_level in enumerate(noise):
    plt.subplot(len(noise), len(kernels) + 1, i*(len(kernels)+1) + 1)
    plt.imshow(A, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    for j, kernel in enumerate(kernels):
        plt.subplot(len(noise), len(kernels) + 1, i*(len(kernels)+1) + j + 2)
        filtA = LSI(A, kernel, np.random.normal(0, noise_level, size=A.shape))
        plt.imshow(filtA, cmap="gray")
        plt.title("K {}, NL {}".format(j+1, noise_level))
        plt.axis("off")
    
plt.tight_layout()
plt.show()


# %% [markdown]
# 2.2

# %%
def restoration(img,psf):
    psfFFT = fft2(psf,s=img.shape)
    imgFFT = fft2(img,axes=(0,1))
    restoredFFT = np.divide(imgFFT,np.conj(psfFFT))
    imgRestored = np.abs(ifft2(restoredFFT))
    return imgRestored

A = imread("monster.jpg",as_gray=True)
A = imread("trui.png",as_gray=True)

psf = gaussian_kernel(9, sig=1)

imgMild = LSI(A,psf,np.random.normal(0, 0, size=A.shape))
imgExtreme = LSI(A,psf,np.random.normal(0, 50, size=A.shape))
imgRestMild = restoration(imgMild,psf)
imgRestExtr = restoration(imgExtreme,psf)



plt.subplot(2,2,1)
plt.imshow(imgMild,cmap="gray"),plt.axis("off"),plt.title("NL=5 original")
plt.subplot(2,2,2)
plt.imshow(imgExtreme,cmap="gray"),plt.axis("off"),plt.title("NL=50 original")
plt.subplot(2,2,3)
plt.imshow(imgRestMild,cmap="gray"),plt.axis("off"),plt.title("Mild original")
plt.subplot(2,2,4)
plt.imshow(imgRestExtr,cmap="gray"),plt.axis("off"),plt.title("Mild original")
plt.tight_layout()
plt.show()

# %% [markdown]
# 2.3

# %%
def weinerFilt(img,psf,K):
    psfFFT = fft2(psf,s=img.shape)
    imgFFT = fft2(img,axes=(0,1))
    sigPower = np.abs(psfFFT)**2

    filter = 1/psfFFT * sigPower / (sigPower + K)
    restoredFFT = filter * imgFFT
    return np.abs(ifft2(restoredFFT))

A = imread("trui.png",as_gray=True)

psf = gaussian_kernel(9)

imgMild = LSI(A,psf,np.random.normal(0, 0.01, size=A.shape))
imgExtreme = LSI(A,psf,np.random.normal(0, 50, size=A.shape))
imgRestMild = weinerFilt(imgMild,psf,K=0.1)
imgRestExtr = weinerFilt(imgExtreme,psf,K=0.1)



plt.subplot(2,2,1)
plt.imshow(imgMild,cmap="gray"),plt.axis("off"),plt.title("NL=5 original")
plt.subplot(2,2,2)
plt.imshow(imgExtreme,cmap="gray"),plt.axis("off"),plt.title("NL=50 original")
plt.subplot(2,2,3)
plt.imshow(imgRestMild,cmap="gray"),plt.axis("off"),plt.title("Mild original")
plt.subplot(2,2,4)
plt.imshow(imgRestExtr,cmap="gray"),plt.axis("off"),plt.title("Mild original")
plt.tight_layout()
plt.show()

# %% [markdown]
# # 3

# %% [markdown]
# 3.4

# %%
def centred_square(x, y, N):
    if x%2==0 or y%2==0:
        raise ValueError("The x and y dimensions of the image must be odd")
    output = np.zeros((x, y))
    x_center, y_center = x//2, y//2
    half_N = N//2
    if N%2==1:
        output[x_center-half_N:x_center+half_N+1, y_center-half_N:y_center+half_N+1] = 1
    else:
        output[x_center-half_N:x_center+half_N, y_center-half_N:y_center+half_N] = 1
    return output

plt.imshow(centred_square(9, 9, 3), cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.show()

# %% [markdown]
# 3.5

# %%
def filter_translation(image, tx, ty, mode="full", boundary="fill"):
    filter_x, filter_y = np.zeros(2*np.abs(tx)+1), np.zeros(2*np.abs(ty)+1)
    filter_x[np.abs(tx)+tx] = 1
    filter_y[np.abs(ty)+ty] = 1
    filter = np.outer(filter_y, filter_x)
    output = convolve2d(image, filter, mode, boundary)
    return output
A = centred_square(9, 9, 3)
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Original image")
B = filter_translation(A, 3, 3, mode="same")
plt.subplot(1, 2, 2)
plt.imshow(B, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Filter translation")
plt.show()

# %% [markdown]
# 3.6

# %%
def homogeneous_translation(image, tx, ty):
        output = np.zeros(image.shape)
        matrix = np.identity(3)
        matrix[0, 2] = tx
        matrix[1, 2] = ty
        matrix = np.linalg.inv(matrix)
        x = np.linspace(1, image.shape[1], image.shape[1])
        y = np.linspace(1, image.shape[0], image.shape[0])
        X, Y = np.meshgrid(x, y)
        for i in range(len(x)):
            for j in range(len(y)):
                loc = matrix@np.array([x[i], y[j], 1])
                x_nearest = int(round(loc[0]-1))
                y_nearest = int(round(loc[1]-1))
                if x_nearest >= image.shape[0] or y_nearest >= image.shape[1]:
                    output[j, i] = 0
                else:    
                    output[j, i] = image[y_nearest,x_nearest]
        return output
    
    
A = centred_square(9, 9, 3)
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Original image")
B = homogeneous_translation(A, 0.6, 1.2)
plt.subplot(1, 2, 2)
plt.imshow(B, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Homogeneous translation")
plt.show()

# %% [markdown]
# 3.7

# %%
def fourier_translation(image, tx, ty):
    fft_image = fft2(image)
    u = np.fft.fftfreq(fft_image.shape[1])
    v = np.fft.fftfreq(fft_image.shape[0])
    u, v = np.meshgrid(u, v)
    rotate = np.exp(-2j*np.pi*(u*tx+v*ty))
    return np.real(ifft2(fft_image*rotate))

A = centred_square(9, 9, 3)
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Original image")
C = fourier_translation(A, 3, 3)
plt.subplot(1, 2, 2)
plt.imshow(C, cmap="gray")
plt.xticks(np.arange(0, 9))
plt.yticks(np.arange(0, 9))
plt.grid(linewidth=0.15, color="blue", alpha=0.75)
plt.title("Fourier translation")
plt.show()

# %% [markdown]
# 3.8

# %%
A1 = centred_square(33, 33, 7)
B = fourier_translation(A1, 5.6, 7.2)
plt.subplot(1, 2, 1)
plt.imshow(B, cmap="gray")
plt.title("Centred white square")
A2 = imread("cameraman.tif")
C = fourier_translation(A2, 15.3, 30.7)
plt.subplot(1, 2, 2)
plt.imshow(C, cmap="gray")
plt.title("cameraman.tif")
plt.show()



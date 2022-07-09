from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt 
import cv2
from skimage import io

img_orig = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')

#Q1
def rgb2bgr(img):
    return img[...,::-1]

#Q2
def rgb2gray(img):
    _img= img.copy().astype(np.float32)
    gray = _img[...,0] * 0.2126 + _img[...,1] * 0.7152 + _img[...,2] * 0.0722
    gray = np.clip(gray,0,255)
    return gray.astype(np.uint8)

img_gray = rgb2gray(img_orig)

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
plt.title('input')
plt.imshow(img_orig)
plt.subplot(1,2,2)
plt.title('answer')
plt.imshow(img_gray,cmap='gray')
plt.show()
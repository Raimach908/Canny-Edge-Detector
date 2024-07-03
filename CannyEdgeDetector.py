import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def applyKernal(image, kernal):
    if len(image.shape) == 3:  # Multi-channel image (e.g., RGB)
        R, G, B = cv.split(image)
        red = convolute(R, kernal)
        green = convolute(G, kernal)
        blue = convolute(B, kernal)
        updated = cv.merge((red, green, blue))
    elif len(image.shape) == 2:  # Single-channel image (e.g., grayscale)
        updated = convolute(image, kernal)
    else:
        raise ValueError("Unsupported image format")
    
    return updated

def convolute(image,kernal):
    height, width = image.shape
    kheight, kwidth = kernal.shape
    padded_array = np.pad(image,pad_width=kwidth // 2)
    resultant_image = np.zeros((height,width),dtype=int)

    for i in range(height):
        for j in range(width):
            mat = padded_array[i:i+kheight,j:j+kwidth]
            if (mat.shape == kernal.shape):
                resultant_image[i,j] = np.sum(mat * kernal)

    return resultant_image  

# Step 1: Smooth Image with Gaussian filter
def Gaussian_Filter(image, sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - center) * 2 + (y - center) * 2) / (2 * sigma ** 2))
    smooth_image = applyKernal(image,kernel)
    smooth_image = smooth_image.astype(np.uint8)
    return smooth_image

# Step 2: Compute Derivative of filtered image 
def findDerivatives(smooth_image):
        # Sobel X Kernel
    sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    # Sobel Y Kernel
    sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
    x_derivative = applyKernal(smooth_image,sobel_x_kernel)
    y_derivative = applyKernal(smooth_image,sobel_y_kernel)
    return x_derivative,y_derivative

# Step 3:  Find Magnitude and Orientation of gradient
def gradient_Calculation(x_derivative,y_derivative):
    magnitude = np.sqrt(x_derivative**2+y_derivative**2)
    angle = np.arctan2(y_derivative,x_derivative)
    return magnitude,angle

# Step 4: Apply Non-max suppression
def non_max_suppression(magnitude,orientation):
    rows, cols = magnitude.shape[:2]
    suppressed = np.zeros_like(magnitude)

    orientation = np.rad2deg(orientation)
    orientation[orientation < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q, r = 255, 255
            angle = orientation[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (angle >= -22.5 and angle < 0):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    return suppressed

# Step 5: Apply Thresholding (Hysteresis)
def apply_threshold(suppressed,low_threshVal,high_threshVal):
    high_threshVal = suppressed.max() * high_threshVal
    low_threshVal = low_threshVal * high_threshVal
    rows, columns = suppressed.shape
    thresholdedVal = np.zeros_like(suppressed)

    strongi,strongj = np.where(suppressed >= high_threshVal)
    weaki,weakj = np.where((suppressed >= low_threshVal) & (suppressed <= high_threshVal))

    thresholdedVal[strongi,strongj] = 255
    thresholdedVal[weaki,weakj] = 0

    return thresholdedVal

#                   <<---i. Implement canny edge detector--->>

path = input("Enter image path: ")
image = cv.imread(path)
if image is None:
    print("Error: Could not read image")
else:
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.resize(gray_image, (450, 450))

    plt.subplot(2, 4, 1), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)), plt.title("Original Image"), plt.xticks([]), plt.yticks([])
    
    smooth_image = Gaussian_Filter(gray_image, sigma=7, kernel_size=3)
    dx, dy = findDerivatives(smooth_image)
    magnitude, orientation = gradient_Calculation(dx,dy)
    suppressed = non_max_suppression(magnitude, orientation)
    thresholded_Image = apply_threshold(suppressed, low_threshVal=0.05, high_threshVal=0.15)


    plt.subplot(2, 4, 2), plt.imshow(smooth_image, cmap='gray'), plt.title("Smoothed Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 3), plt.imshow(dx, cmap='gray'), plt.title("dx"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 4), plt.imshow(dy, cmap='gray'), plt.title("dy"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 5), plt.imshow(magnitude, cmap='gray'), plt.title("Magnitude"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 6), plt.imshow(orientation, cmap='gray'), plt.title("Orientation"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 7), plt.imshow(suppressed, cmap='gray'), plt.title("suppressed Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 8), plt.imshow(thresholded_Image, cmap='gray'), plt.title("Canny Edge Detector"), plt.xticks([]), plt.yticks([])

    plt.show()
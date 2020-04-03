# preprocessMethod.py
# Runs a preprocessing method against the intermediate dataset and outputs them
# into the "preprocessed" folder

# Binarization - Adaptive Thresholding
# Adaptive thresholding is used instead of Simple thresholding because the image may have different lighting conditions
# in different areas.
# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

import cv2 as cv
import numpy as np
import time
import os
from matplotlib import pyplot as plt


def runPreprocess(image_dir):
    imageCount = 100
    images = []
    tempImages = [] 
    times = []
    
    # Load intermediate set into images list
    for i in range(imageCount):
        images.append(str(image_dir) + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg")
        
        
    # Preprocess the images and store them in a temp list
    for i in range(len(images)):
        startTime = int(round(time.time() * 1000))
        # Open the image file in greyscale (0)
        tempImage = cv.imread(images[i], 0)
        # Blurring (averaging)/Smoothing the image to reduce noise
        #Maybe not use this??
        tempImage = cv.medianBlur(tempImage, 5)
        
        # Do the preprocessing stuff here
        # Apply Adaptive thresholding method - Guassian threshold
        tempImage = cv.adaptiveThreshold(tempImage, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        
        # The preprocesed images are saved temporarily in memory instead of written into output directory
        # so calculating the actual processing time won't be affected
        tempImages.append(tempImage)
        
        # Record elapsed processing time for the image
        times.append(int(round(time.time() * 1000)) - startTime)
        
    print("Total processing time: ", sum(times), "ms")
    print("Average processing time: ", sum(times)/len(times), "ms")
        
    return tempImages
        

def main():
    image_dir = "intermediate"
    
    # Preprocess the images
    processedImages = runPreprocess(image_dir)
    
    
    # Output processed images into output directory
    output_dir = "binarized"
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
        
    for i in range(len(processedImages)):
        tempImage = processedImages[i]
        cv.imwrite(output_dir + "/W2_XL_input_noisy_" + str(1000 + i) + ".jpg", tempImage)
        
    print("Saved processed images to binarized directory")

main()

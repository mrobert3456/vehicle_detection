import cv2 as cv
import numpy as np


def get_image_roi(image):
    mask = np.zeros_like(image)
    im_shape = image.shape

    vertices = np.array(
        [[
          (0, im_shape[0] * .85),# bottom left
          (im_shape[1] * .10, im_shape[0] * .35), # top left
          (im_shape[1] * .90, im_shape[0] * .35), # top right
          (im_shape[1], im_shape[0] * .85) # bottom right
        ]], dtype=np.int32)  # creates an array with the trapezoids verticies

    cv.fillPoly(mask, vertices, (255,) * 3)
    masked_image = cv.bitwise_and(image, mask)  # crops the original image with the mask

    return masked_image
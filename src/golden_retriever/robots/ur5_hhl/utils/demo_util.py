# Adopted from code by Dian, Xupeng, Haojie, Mingxi

import numpy as np

upper_right = [-0.53132435, 0.25073409]
down_right = [-0.18128785, 0.25072697]
upper_left = [-0.53324759, -0.23809865]
down_left = [-0.18124463, -0.23811645]
UPPER_XY = -0.53132435
DOWN_XY = -0.18128785
LEFT_XY = -0.23809865
RIGHT_XY = 0.25072697
UPPERDOWN_XY = down_right[0] - upper_left[0]
LEFTRIGHT_XY = down_right[1] - down_left[1]
Z_MIN_ROBOT = -0.55862

# pixel_x_reso = 208
# pixel_y_reso = 288
pixel_x_reso = 206
pixel_y_reso = 284

UPPER_PIX = 35
DOWN_PIX = 243
LEFT_PIX = 141 + 3
RIGHT_PIX = 436 - 4
UPPERDOWN_PIXEL = DOWN_PIX - UPPER_PIX
LEFTRIGHT_PIXEL = RIGHT_PIX - LEFT_PIX


def xy2pixel(xy):
    # upper left corner of the action space is the pixel origin
    x, y = xy
    pixel_x = pixel_x_reso * (x - UPPER_XY) / UPPERDOWN_XY
    pixel_y = pixel_y_reso * (y - LEFT_XY) / LEFTRIGHT_XY
    pixel = np.array([pixel_x, pixel_y]).astype(int)
    return pixel


def pixel2xy(pixel):
    pixel_x, pixel_y = pixel
    x = UPPER_XY + UPPERDOWN_XY * pixel_x / pixel_x_reso
    y = LEFT_XY + LEFTRIGHT_XY * pixel_y / pixel_y_reso
    xy = np.array([x, y])
    return xy


def pixel2xyz(pixel):
    x, y = pixel2xy(pixel)
    z = -0.1511397
    return (x, y, z)


def getGraspZ(pixel, depthmap):
    # take pixel_xy and output z in robot coordinate
    Z_MIN_DEPTH_PIXEL = 1.051
    height = Z_MIN_DEPTH_PIXEL - depthmap[pixel] / 1000
    return Z_MIN_ROBOT + height


if __name__ == "__main__":
    print(1)

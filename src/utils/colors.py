import numpy as np


def rgb2yuv(r, g, b, mode="ycbcr"):
    # 8 bit full scale Y Cb Cr
    Y = [0.299, 0.587, 0.114]
    U = [-0.169, -0.331, 0.5]
    V = [0.5, -0.419, -0.081]
    yuv = np.asarray([Y, U, V])

    if mode == "ycbcr":
        return yuv.dot(np.asarray([r, g, b]))
    elif mode == "yuv":
        return yuv.dot(np.asarray([r, g, b])) - np.array([0, 128.0, 128.0])
    else:
        return None


def yuv2rgb(y, u, v):
    r = [1.0, 0.0, 1.402]
    g = [1.0, -0.344, -0.714]
    b = [1.0, 1.772, 0.0]
    rgb = np.asarray([r, g, b])
    return rgb.dot(np.asarray([[y, u, v]]))


def detectColormode(mode):
    if mode == "RGB" or mode == "RGBY":
        return "rgb"
    elif mode == "YUV" or mode == "YCbYr":
        return "yuv"
    else:
        raise ValueError("Unknow color mode: {}".format(mode))

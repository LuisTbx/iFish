import imageio
import numpy as np
from math import sqrt
import sys
import argparse
import os
import timeit


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))


def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        # print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            # get xn and yn distance from normalized center
            rd = sqrt(xnd**2 + ynd**2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)


def fish_vectorized(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """
    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        # print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # Array with coordinates, normalize between -1 and 1
    cord_x, cord_y = np.meshgrid(np.arange(w), np.arange(h))
    xnd_ = (cord_x.astype(float)*2 - w).T / w
    ynd_ = (cord_y.astype(float)*2 - h).T / h

    rd_ = np.sqrt(xnd_ ** 2 + ynd_ ** 2)
    rd = (1 - (distortion_coefficient * (rd_ ** 2)))
    # Find zero values to clean them later
    wrong_radius = np.where(rd == 0)

    # Add epsilon to avoid division by 0.
    rd[wrong_radius] += np.finfo(float).eps
    xdu_, ydu_ = xnd_ / rd, ynd_ / rd

    # If zero values were found, assign the original value
    if len(wrong_radius[0]) > 0:
        xdu_[wrong_radius[0], wrong_radius[1]] = xnd_[wrong_radius[0], wrong_radius[1]]
        ydu_[wrong_radius[0], wrong_radius[1]] = ynd_[wrong_radius[0], wrong_radius[1]]

    # Convert the centered coordinates back to pixel size
    xu_, yu_ = ((xdu_ + 1) * w) / 2, ((ydu_ + 1) * h) / 2
    xu_ = xu_.astype(int)
    yu_ = yu_.astype(int)

    # Identify all spurious indexes smaller than zero, bigger than h or w
    temp_index1 = xu_ >= 0
    temp_index2 = xu_ < w
    temp_index3 = yu_ >= 0
    temp_index4 = yu_ < h

    # Get valid only indices
    valid_indices = (temp_index1 & temp_index2) & (temp_index3 & temp_index4)

    # Mask values
    yu_ *= valid_indices
    xu_ *= valid_indices

    # Select valid values
    valid = np.where(valid_indices)
    dstimg[valid] = img[xu_[valid], yu_[valid]]

    return dstimg.astype(np.uint8)


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Apply fish-eye effect to images.",
        prog='python3 fish.py')

    parser.add_argument("-i", "--image", help="path to image file."
                        " If no input is given, the supplied example 'grid.jpg' will be used.",
                        type=str, default="grid.jpg")

    parser.add_argument("-o", "--outpath", help="file path to write output to."
                        " format: <path>.<format(jpg,png,etc..)>",
                        type=str, default="fish.png")

    parser.add_argument("-d", "--distortion",
                        help="The distortion coefficient. How much the pixels move from/to the center."
                        " Recommended values are between -1 and 1."
                        " The bigger the distortion, the further pixels will be moved outwars from the center (fisheye)."
                        " The smaller the distortion, the closer pixels will be move inwards toward the center (rectilinear)."
                        " For example, to reverse the fisheye effect with --distortion 0.5,"
                        " You can run with --distortion -0.3."
                        " Note that due to double processing the result will be somewhat distorted.",
                        type=float, default=0.5)

    return parser.parse_args(args)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == "__main__":
    args = parse_args()
    try:
        imgobj = imageio.imread(args.image)
    except Exception as e:
        print(e)
        sys.exit(1)
    if os.path.exists(args.outpath):
        ans = input(
            args.outpath + " exists. File will be overridden. Continue? y/n: ")
        if ans.lower() != 'y':
            print("exiting")
            sys.exit(0)

    wrapped_original = wrapper(fish, imgobj, args.distortion)
    wrapped_vec = wrapper(fish_vectorized, imgobj, args.distortion)

    t1 = timeit.timeit(wrapped_original, number=100)
    t2 = timeit.timeit(wrapped_vec, number=100)
    print("Time loop implementation: {}, time vectorization {}".format(t1, t2))
    print("Image size: {}".format(imgobj.shape))
    output_img = fish_vectorized(imgobj, args.distortion)
    imageio.imwrite(args.outpath, output_img, format='png')

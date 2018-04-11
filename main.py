import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw

# window size
DELTA_X = 5
DELTA_Y = 5
# trace scaling factor
K = 0.04
# detection threshold
CORNER_THRESHOLD = 1e10
# color of the crosses used to indicate a corner
CROSS_COLOR = (255, 0, 0, 255)


def draw_cross(draw, x, y):
    """
    Draws a cross at the coordinates x, y using the draw instance.
    """
    l = 5
    draw.line((x, y - l, x, y + l), fill=CROSS_COLOR)
    draw.line((x - l, y, x + l, y), fill=CROSS_COLOR)


def grey_scale_pil(image):
    """
    Converts image to grey scale by invoking PIl
    """
    im = Image.fromarray(image)
    return np.array(im.convert("L"))


def grey_scale(image):
    """
    Converts a colour image to a grey scale image
    """
    grey_image = np.zeros(image.shape[0:2])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            grey_image[i, j] = 0.21*pixel[0] + 0.72*pixel[1] + 0.07*pixel[2]
    return grey_image


def calc_derivatives(image):
    """
    Calculates and returns the products of the derivatives.
    i.e. Ixx, Ixy, Iyy
    """
    # first order derivatives
    Iy, Ix = np.gradient(image)
    # product of derivatives
    Ixx = Ix**2.0
    Iyy = Iy**2.0
    Ixy = Ix * Iy
    return Ixx, Ixy, Iyy


def calc_tensor(Ixx, Ixy, Iyy, x, y):
    """
    Calculates and returns the structure tensor M.
    """
    # sum over window
    Sxx = sum(Ixx[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].ravel())
    Sxy = sum(Ixy[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].ravel())
    Syy = sum(Iyy[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].ravel())
    M = np.asarray([[Sxx, Sxy], [Sxy, Syy]])
    return M


def harris_corner_detection(grey_scale_image):
    """
    Performs harris corner detection for image.
    Returns a list of corners.
    """
    print("Calculating derivatives...")
    # im = Image.fromarray(grey_scale_image)
    # im.show()
    Ixx, Ixy, Iyy = calc_derivatives(grey_scale_image)
    size_x = grey_scale_image.shape[0]
    size_y = grey_scale_image.shape[1]
    print("Started corner detection...")
    print("Calculating r_values for all pixels...")
    r_values = np.zeros(grey_scale_image.shape)
    # for each pixel which does not cause an out of range exception
    for x in range(DELTA_X, size_x - DELTA_X):
        for y in range(DELTA_Y, size_y - DELTA_Y):
            M = calc_tensor(Ixx, Ixy, Iyy, x, y)
            r = np.linalg.det(M) - K*np.trace(M)**2.0
            r_values[x, y] = r
    print("Thresholding and nonmax supression...")
    list_of_corners = []
    for x in range(DELTA_X, size_x - DELTA_X):
        for y in range(DELTA_Y, size_y - DELTA_Y):
            # thresholding and nonmax supression
            max_in_window = max(r_values[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].flatten())
            if (r_values[x, y] > CORNER_THRESHOLD) and r_values[x, y] == max_in_window:
                list_of_corners.append((x, y))
    return list_of_corners


def patch_vectors(corner_list, image):
    patches = []
    for i, (x, y) in enumerate(corner_list):
        v = image[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].flatten()
        norm = np.linalg.norm(v, 2)
        v /= norm
        patches.append(v)
    return np.vstack(patches)


def match_images(images, show_corners):
    corner_lists = []
    M = []
    print(show_corners)
    for i, image in enumerate(images):
        print("Converting to grey scale...")
        grey_scale_image = grey_scale(image)
        corner_lists.append(harris_corner_detection(grey_scale_image))
        print("Found {} corners in image {}.".format(len(corner_lists[-1]), i + 1))
        if show_corners:
            print("Coloring corners...")
            im = Image.fromarray(image)
            draw = ImageDraw.Draw(im)
            for (x, y) in corner_lists[-1]:
                draw_cross(draw, y, x)
            im.show()
        M.append(patch_vectors(corner_lists[-1], grey_scale_image))
    R = M[0] * M[1].T

def main():
    """
    Main function.
    """
    # for testing
    sys.argv = ["main.py", "--file1", "arch1.png", "--file2", "arch2.png"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="", help="First file for matching.")
    parser.add_argument("--file2", type=str, default="", help="Second file for matching.")
    parser.add_argument("--show_corners", default=False, action="store_true",
                        help="If set will show the detected corners for both images.")
    args = parser.parse_args()
    if args.file1 == "" or args.file2 == "":
        print("Please specify both files.")
    else:
        # load image into 3d array
        image1 = np.asarray(Image.open(args.file1))
        print("Read {0}x{1} image.".format(*image1.shape))
        image2 = np.asarray(Image.open(args.file2))
        print("Read {0}x{1} image.".format(*image2.shape))
        match_images([image1, image2], args.show_corners)

if __name__ == "__main__":
    main()

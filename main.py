import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw

from exhaustive_ransac import basic_RANSAC

# window size
DELTA_X = 10
DELTA_Y = 10
# trace scaling factor
K = 0.06
# detection threshold
CORNER_THRESHOLD = 1e5
# detection multiplier for max_r value
CORNER_THRESHOLD_MULTIPLIER = 0.1
# color of the crosses used to indicate a corner
CROSS_COLOR = (255, 0, 0, 255)


def draw_cross(draw, x, y):
    """
    Draws a cross at the coordinates x, y using the draw instance.
    """
    l = 5
    draw.line((x, y - l, x, y + l), fill=CROSS_COLOR)
    draw.line((x - l, y, x + l, y), fill=CROSS_COLOR)


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
    Returns a list of corners with inverted coordinates i.e. (x, y) corresponds
    to row x and column y in the image.
    """
    print("Started corner detection...")
    size_x = grey_scale_image.shape[0]
    size_y = grey_scale_image.shape[1]
    print("Calculating derivatives...")
    Ixx, Ixy, Iyy = calc_derivatives(grey_scale_image)
    print("Calculating r_values for all pixels...")
    r_values = np.zeros(grey_scale_image.shape)
    # for each pixel which does not cause an out of range exception
    for x in range(DELTA_X, size_x - DELTA_X):
        for y in range(DELTA_Y, size_y - DELTA_Y):
            # calculate the r_value i.e. the harris corner function
            M = calc_tensor(Ixx, Ixy, Iyy, x, y)
            r = np.linalg.det(M) - K*np.trace(M)**2.0
            # save it the responses into an array
            r_values[x, y] = r
    print("Thresholding and nonmax supression...")
    max_r = max(r_values.flatten())
    list_of_corners = []
    # thresholding and nonmax supression
    for x in range(DELTA_X, size_x - DELTA_X):
        for y in range(DELTA_Y, size_y - DELTA_Y):
            max_in_window = max(r_values[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].flatten())
            # only use those r_values that are bigger than the threshold
            # and are the maximum in their respective window
            # if both are filfilled we found a corner
            if (r_values[x, y] >= max_r * CORNER_THRESHOLD_MULTIPLIER) and (r_values[x, y] == max_in_window):
                list_of_corners.append((x, y))
    # return all found corners
    return list_of_corners


def patch_vectors(corner_list, image):
    """
    Calculates the M matrix containing the patch vector for each corners as a row vector.
    """
    patches = []
    for i, (x, y) in enumerate(corner_list):
        v = image[x - DELTA_X: x + DELTA_X + 1, y - DELTA_Y: y + DELTA_Y + 1].flatten()
        v -= np.mean(v)
        norm = np.linalg.norm(v, 2)
        v /= norm
        patches.append(v)
    return np.vstack(patches)

def draw_images_with_offset(im1, im2, row_offset, column_offset):
    """
    Draws the 2 images overlapping using the given offsets.
    """
    c_o = int(column_offset)
    r_o = int(row_offset)
    abs_c_o = abs(int(column_offset))
    abs_r_o = abs(int(row_offset))
    if len(im1.shape) > 2:
        new_image = np.zeros((im1.shape[0] + 2*abs_c_o,
                              im1.shape[1] + 2*abs_r_o, im1.shape[2]))
        mode = "RGBA"
    else:
        new_image = np.zeros((im1.shape[0] + 2*abs_c_o,
                              im1.shape[1] + 2*abs_r_o))
        mode = "L"
    new_image[abs_c_o: abs_c_o + im1.shape[0], abs_r_o: abs_r_o + im1.shape[1]] = im1
    new_image[abs_c_o + c_o: abs_c_o + c_o + im2.shape[0],
              abs_r_o + r_o: abs_r_o + r_o + im2.shape[1]] = im2
    new_image = np.uint8(new_image)
    im = Image.fromarray(new_image, mode=mode)
    im.show()

def match_images(images, show_corners):
    """
    Matches the 2 images and displays the result.
    Will show the results of the harris corner detection for each image if
    show_corners is set to true.
    """
    corner_lists = []
    M = []
    # this loop detects the corners and calculates the patch vectors for each image
    for i, image in enumerate(images):
        print("### Image {} ###".format(i + 1))
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
        print("Calculating patch vectors")
        M.append(patch_vectors(corner_lists[-1], grey_scale_image))
    print("### Matching images ###")
    # calculate match matrix
    R = np.dot(M[0], M[1].T)
    # filtering i.e. only matches over 0.9 and best possible match per row
    matches = []
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            if R[y, x] == max(R[y]) and R[y, x] > 0.9:
                matches.append((y, x))
    print("Doing RANSAC...")
    # reverse the coordinates in the lists for RANSAC
    list1 = [(y, x) for (x, y) in corner_lists[0]]
    list2 = [(y, x) for (x, y) in corner_lists[1]]
    row_offset, column_offset, match_count = basic_RANSAC(matches, list1, list2)
    # final step: display the result
    print("Result: {} {}".format(row_offset, column_offset))
    print("Drawing images...")
    draw_images_with_offset(images[0], images[1], row_offset, column_offset)


def main():
    """
    Main function.
    """
    # for testing
    # sys.argv = ["main.py", "--file1", "arch1.png", "--file2", "arch2.png"]
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
        print("Read image 1: {0}x{1} pixels.".format(*image1.shape))
        image2 = np.asarray(Image.open(args.file2))
        print("Read image 2: {0}x{1} pixels.".format(*image2.shape))
        match_images([image1, image2], args.show_corners)
        # draw_images_with_offset(image1, image2, 0, 0)

if __name__ == "__main__":
    main()

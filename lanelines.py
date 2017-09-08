import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


import tkinter



# prepare object points
nx = 9
ny = 6

debug = False


# TODO Make a list of calibration images, instead of single image
fname = 'camera_cal/calibration1.jpg'
fnames = glob.glob('camera_cal/calibration*.jpg')

# Initially copied from course materials, "9. Finding Corners"
def drawCorners(fname):
    img = cv2.imread(fname)
    cv2.imshow('input-img', img)
    cv2.waitKey(500)
    if img == None:
        raise ValueError("Image {} not found".format(fname))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret:
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        if (debug):
            cv2.imshow('cornerimg', img)
            cv2.waitKey(500)
        return img
    print("No corners found in {}".format(fname))
    return None


# initially copied from course materials, "10 - Calibrating your camera"
def accumulate_calibration(img, imgpoints, objpoints):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (7,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)  # x, y coordinates

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # for jupyter notebook: plt.imshow(img)

        # For cv2-based image display
        if (debug):
            cv2.imshow('chessboardcorners', img)
            cv2.waitKey(500)
    return


# TODO this should run after all {obj,img}points have been found, on (road)images to correct
def image_calibration_params(img, objpoints, imgpoints):
    gray_shape = img.shape[0:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    return mtx, dist


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


def show_distorted_and_undistorted(an_image, mtx, dist):
    undistorted1 = undistort_image(an_image, mtx, dist)
    cv2.imshow("undistorted", undistorted1)
    cv2.imshow("distorted", an_image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def main():
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for fname in fnames:
        # Display the image, with corner-dots but without calibration
        # drawCorners(fname)

        # Accumulate calibration data
        img = cv2.imread(fname)
        accumulate_calibration(img, imgpoints, objpoints)

    an_image = cv2.imread(fnames[0])
    mtx, dist = image_calibration_params(an_image, objpoints, imgpoints)
    print("Calibration parameters: \nmtx={}, \ndist={}".format(mtx, dist))

    #show_distorted_and_undistorted(an_image, mtx, dist)
    #show_distorted_and_undistorted(cv2.imread("test_images/test6.jpg"), mtx, dist)



main()

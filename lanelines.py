import numpy as np
import cv2
import glob
# importing matplotlib.pyplot causes TK to crash on init :(


# prepare object points
nx = 9
ny = 6

debug = False


# TODO Make a list of calibration images, instead of single image
calibration_f_name = 'camera_cal/calibration1.jpg'
calibration_f_names = glob.glob('camera_cal/calibration*.jpg')

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


# should run after all {obj,img}points have been found, on (road)images to correct
def image_calibration_params(img, objpoints, imgpoints):
    gray_shape = img.shape[0:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    return mtx, dist

def calibration_params(imgpoints, objpoints):
    for fname in calibration_f_names:
        # Display the image, with corner-dots but without calibration
        # drawCorners(fname)

        # Accumulate calibration data
        img = cv2.imread(fname)
        accumulate_calibration(img, imgpoints, objpoints)
    an_image = cv2.imread(calibration_f_names[0])
    mtx, dist = image_calibration_params(an_image, objpoints, imgpoints)
    return dist, mtx


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def show_distorted_and_undistorted(an_image, mtx, dist):
    undistorted1 = undistort_image(an_image, mtx, dist)
    cv2.imshow("undistorted", undistorted1)
    cv2.imshow("distorted", an_image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


# Originally copied from 30. Color and Gradient
# assume img in BGR format
def threshold_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary


# Originally copied from "15. Transform a Stop Sign"
def dashboard_to_overhead(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])  # keep same size as input image
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped



def lanelines_main():
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    dist, mtx = calibration_params(imgpoints, objpoints)
    #print("Calibration parameters: \nmtx={}, \ndist={}".format(mtx, dist))
    #show_distorted_and_undistorted(an_image, mtx, dist)
    #show_distorted_and_undistorted(cv2.imread("test_images/test6.jpg"), mtx, dist)


    # TODO threshold image by combining sobel ops + color space conversion
    original_img = cv2.imread("test_images/straight_lines1.jpg")
    undistorted_img = undistort_image(original_img, mtx, dist)
    #img = cv2.imread("test_images/straight_lines1.jpg")
    color_binary, combined_binary = threshold_pipeline(undistorted_img)

    #cv2.imshow("bluegreen-thresholds", color_binary)
    #cv2.imshow("combined-thresholds", combined_binary)
    #cv2.waitKey(20000)

    height, width = undistorted_img.shape[:2]
    # Origin is top left corner, y increases downwards
    # source below goes (too?) far ahead almost to horizon
    src = np.float32([[610, 441], [669, 441], [258, 682], [1049, 682]])
    dst = np.float32([[450, 0], [width - 450, 0], [450, height], [width-450, height]])



    overhead_img = dashboard_to_overhead(color_binary, src, dst)

    cv2.imshow("undistorted", undistorted_img)
    cv2.imshow("bluegreen-thresholds", color_binary)
    cv2.imshow("overhead", overhead_img)

    cv2.waitKey(200000)

    # TODO watch walkthrough from around 26:29 and forward:
    # https://www.youtube.com/watch?v=vWY8YUayf9Q&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=4
    cv2.destroyAllWindows()


if __name__ == "__main__":
    lanelines_main()

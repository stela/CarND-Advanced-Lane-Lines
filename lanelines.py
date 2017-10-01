import numpy as np
import cv2
import glob
# importing matplotlib.pyplot causes TK to crash on init :(
import matplotlib.pyplot as plt

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


# Originally from "15. Transform a Stop Sign"
def dashboard_to_overhead(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])  # keep same size as input image
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped


# Originally from "33. Finding the Lines"
def find_lane_lines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    bottom_half_y = np.int(binary_warped.shape[0]/2)
    histogram = np.sum(binary_warped[bottom_half_y:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    quarter_point = np.int(midpoint/2)
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:midpoint+quarter_point]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Add rectangular windows indicating where histogram-lane-search was done
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


#    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


# def visualize_lane_lines(out_img, binary_warped, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img = draw_lane_polynomial(out_img, left_fitx, right_fitx, margin, ploty)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img, ploty, left_fitx, right_fitx


# Googled for cv2.fillPoly/cv2.polylines usage and found this function (slightly modified) at
# https://github.com/rioffe/CarND-Advanced-Lane-Lines-Solution
#
# Sort of like
# plt.plot(left_fitx, ploty, color='yellow')
# plt.plot(right_fitx, ploty, color='yellow')
# but for cv2
#
# Draw a thick polynomial to show curvature of the detected lines
# And recast the x and y points into usable format for cv2.fillPoly()
# since rubric seems to require this
def draw_lane_polynomial(out_img, left_fitx, right_fitx, margin, ploty):
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin / 2, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin / 2, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin / 2, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin / 2, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    window_img = np.zeros_like(out_img)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1.0, window_img, 0.3, gamma=1.0)
    return out_img


# Measuring radius of curvature, originally from course materials
# "35. Measuring Curvature"
def radius_of_curvatute(ploty, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # Example values: 632.1 m    626.2 m

    return left_curverad, right_curverad


def lanelines_main():
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    dist, mtx = calibration_params(imgpoints, objpoints)
    #print("Calibration parameters: \nmtx={}, \ndist={}".format(mtx, dist))
    #show_distorted_and_undistorted(an_image, mtx, dist)
    #show_distorted_and_undistorted(cv2.imread("test_images/test6.jpg"), mtx, dist)


    # TODO threshold image by combining sobel ops + color space conversion
    # original_img = cv2.imread("test_images/straight_lines1.jpg")
    original_img = cv2.imread("test_images/test5.jpg")
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

#    binary_warped = dashboard_to_overhead(color_binary, src, dst)
    binary_warped = dashboard_to_overhead(combined_binary, src, dst)

    # Find the left and right lane lines and their parameters
#    out_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy = find_lane_lines(binary_warped)

    # Mark left/right lines with colors, calculate left/right fit polynomials
    out_img, ploty, left_fitx, right_fitx = \
        find_lane_lines(binary_warped)
#       visualize_lane_lines(out_img, binary_warped, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds)

    left_curverad, right_curverad = radius_of_curvatute(ploty, left_fitx, right_fitx)
    print("left radius: {} right radius: {}".format(left_curverad, right_curverad))

    #cv2.imshow("undistorted", undistorted_img)
    #cv2.imshow("bluegreen-thresholds", color_binary)
    #cv2.imshow("overhead", binary_warped)
    cv2.imshow("out_img", out_img)


    cv2.waitKey(200000)

    # TODO watch walkthrough from around 26:29 and forward:
    # https://www.youtube.com/watch?v=vWY8YUayf9Q&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=4
    cv2.destroyAllWindows()


if __name__ == "__main__":
    lanelines_main()

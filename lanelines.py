import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from functools import partial

# prepare object points
nx = 9
ny = 6

debug = False

calibration_f_names = glob.glob('camera_cal/calibration*.jpg')

# Number of sliding windows chosen
nwindows = 9


# Initially copied from course materials, "9. Finding Corners"
def draw_corners(fname):
    img = cv2.imread(fname)
    cv2.imshow('input-img', img)
    cv2.waitKey(500)
    if img is None:
        raise ValueError("Image {} not found".format(fname))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret:
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        if debug:
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


def calibration_params():
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

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


# Originally copied from 30. Color and Gradient
# assume img in BGR format
def threshold_pipeline(img, sobel_x_thresh=(55, 200), luv_l_thresh=(225, 255),
                       lab_l_thresh=(185, 255), lab_b_thresh=(146, 255)):
    img = np.copy(img)
    # Tweaked thresholding according to first reviwer's suggestions for improvements:
    # White is detected well with L of LUV color space [225, 255]
    # Yellow is detected well with b channel of Lab [155, 255] to [200, 255]

    # Then did some own experimentation and found these values to be better (use both L and B of LAB)
    # yellow_L = 185 -
    # yellow_A = 0 -
    # yellow_B = 146 -

    # white_L = 215 or 225 -
    # white_U = 0 -
    # white_V = 0 -

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV).astype(np.float)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float)

    # Self/course: Sobel of HLS's L channel (x-derivative) to detect vertical (lane) lines
    hls_l_channel = hls[:,:,1]
    sobelx = cv2.Sobel(hls_l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient (green in color_binary)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_x_thresh[0]) & (scaled_sobel <= sobel_x_thresh[1])] = 1

    # Thresholding LUV L channel to [225, 255] aimed at white lane lines (red in color_binary)
    luv_l_channel = luv[:,:,0]
    luv_l_binary = np.zeros_like(luv_l_channel)
    luv_l_binary[(luv_l_channel >= luv_l_thresh[0]) & (luv_l_channel <= luv_l_thresh[1])] = 1

    # Thresholding Lab b channel to [155, 200] aimed at yellow lane lines (blue in color_binary)
    lab_l_channel = lab[:,:,0]
    lab_b_channel = lab[:,:,2]
    lab_binary = np.zeros_like(lab_b_channel)
    lab_binary[(lab_l_channel >= lab_l_thresh[0]) & (lab_l_channel <= lab_l_thresh[1])
               & (lab_b_channel >= lab_b_thresh[0]) & (lab_b_channel <= lab_b_thresh[1])] = 1

    # Stack each channel (output is BGR format)
    # zero-channel can be inserted with: np.zeros_like(binaryimage)
    color_binary = np.dstack((lab_binary, sxbinary, luv_l_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (luv_l_binary == 1) | (lab_binary == 1)] = 1

    return color_binary, combined_binary


# Originally from "15. Transform a Stop Sign"
def dashboard_to_overhead(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])  # keep same size as input image
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return M, warped


class LaneLines:
    def __init__(self, ploty, left_fitx, right_fitx, left_lane_centers, right_lane_centers):
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.left_lane_centers = left_lane_centers
        self.right_lane_centers = right_lane_centers


# Originally from "33. Finding the Lines"
def find_lane_lines(binary_warped, previous_lane_lines):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    bottom_half_y = np.int(binary_warped.shape[0]/2)
    histogram = np.sum(binary_warped[bottom_half_y:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    quarter_point = np.int(midpoint/2)
    # Initial left/right window positions based on average across bottom half of the image, not just bottom-most part
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:midpoint+quarter_point]) + midpoint

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
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_lane_centers = []
    right_lane_centers = []

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
        # good_{left,right}_inds are indexes of nonzero pixels inside the window,
        # of the binary_warped.nonzero() collection
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            left_lane_centers.append(leftx_current)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            right_lane_centers.append(rightx_current)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # If nothing detected, quick-and-dirty bogus response
    # Ideally info from previous frames should be used instead
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
        if previous_lane_lines is None:
            left_fitx = np.full(len(ploty), midpoint - 175)
            right_fitx = np.full(len(ploty), midpoint + 175)
            return out_img, ploty, left_fitx, right_fitx, left_lane_centers, right_lane_centers
        else:
            return out_img, previous_lane_lines

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # A "normally curvy" lane line will have:
    # |left/right_fit[2]| < 1000
    # |left/right_fit[1]| < 1 and
    # |left/right_fit[0]| < 0.001
    if abs(left_fit[2]) > 1000 or abs(left_fit[1]) > 1 or abs(left_fit[0]) > 0.1 or \
            abs(right_fit[2]) > 1000 or abs(right_fit[1]) > 1 or abs(right_fit[0]) > 0.1:
        return out_img, previous_lane_lines

    # Generate x and y values for plotting. ploty is just a range from 0 to height of binary_warped
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img = draw_lane_polynomial(out_img, left_fitx, right_fitx, margin, ploty)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # out_img can be displayed for debugging/visualization purposes
    return out_img, LaneLines(ploty, left_fitx, right_fitx, left_lane_centers, right_lane_centers)


# From "33. Finding the Lines - Visualization" with modifications
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
def radius_of_curvature(ploty, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    # According to https://mutcd.fhwa.dot.gov/pdfs/millennium/06.14.01/3ndi.pdf 3A.06
    # (which seems to apply here), rural highway lane markings occur at 40ft/12m intervals
    # and lane lines each are (also according to 35. Measuring Curvature") 10ft/3m.
    # A lane line of 3 m is about 130 pixels long
    # Lane lines are 3.7 m which is about 400 pixels apart
    ym_per_pix = 3./130.   # meters per pixel in y dimension
    xm_per_pix = 3.7/400.  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fit*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Radius of curvature is in meters
    #
    # Using "test2.jpg" which seems to be the location with 1 km radius, I get left: 290 m and right: 423 m
    # Close enough to 1 km I guess ;-)
    return left_curverad, right_curverad


def sideways_offset_lane_center(width, left_centers, right_centers):
    """
    Find sideways-offset to center of lane.
    
    See "36. Tips and Tricks for the Project" - "Offset" for the algorithm

    :param width: image width in pixels
    :type width: int
    :param left_centers: per-window array of left lane line centers, from find_lane_lines()
    :type left_centers: list of int
    :param right_centers: per-window array of right lane line centers, from find_lane_lines()
    :type right_centers: list of int
    :return: number of meters off-center
    :rtype: float
    """
    camera_center = width/2

    # in case nothing at all detected, abort to avoid index out of bounds
    if len(left_centers) == 0 or len(right_centers) == 0:
        return 0

    bottom_left = left_centers[0]
    bottom_right = right_centers[0]
    lanes_center = np.mean([bottom_left, bottom_right])
    # use same conversion as radius_of_curvature() above
    meters_sideways_offset = (lanes_center - camera_center) * (3.7/400.)
    return meters_sideways_offset


# Project polynomials onto original image
# Originally from course materials "36. Tips and Tricks for the Project" but modified
def project_onto_original(undistorted_img, warped, ploty, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
    return result


last_lane_lines = None


# Process each video frame
def process_image(original_img, mtx, dist, conv_rgb_to_bgr=True):
    if conv_rgb_to_bgr:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    # work on the image in BGR space, as if (video) frames were read in using cv2

    undistorted_img = undistort_image(original_img, mtx, dist)
    color_binary, combined_binary = threshold_pipeline(undistorted_img)

    src, dst = transform_src_and_dst(undistorted_img)

    # M, binary_warped = dashboard_to_overhead(color_binary, src, dst)
    M, binary_warped = dashboard_to_overhead(combined_binary, src, dst)

    # Find the left and right lane lines and their parameters, calculate left/right fit polynomials
    global last_lane_lines
    out_img_annotated, lane_lines = \
        find_lane_lines(binary_warped, last_lane_lines)
    last_lane_lines = lane_lines

    # Road radius and car's sideways offset
    left_curverad, right_curverad = radius_of_curvature(lane_lines.ploty, lane_lines.left_fitx, lane_lines.right_fitx)
    meters_sideways_offset = \
        sideways_offset_lane_center(binary_warped.shape[1], lane_lines.left_lane_centers, lane_lines.right_lane_centers)

    # Overlay original image with mask of area between lane lines, print radius and sideways offset
    Minv = np.linalg.inv(M)
    original_img_overlaid =\
        project_onto_original(undistorted_img, binary_warped,
                              lane_lines.ploty, lane_lines.left_fitx, lane_lines.right_fitx, Minv)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_img_overlaid,'Offset: %f m' %(meters_sideways_offset), (33, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(original_img_overlaid,'Left radius: %.1f m' %(left_curverad), (33, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(original_img_overlaid,'Right radius: %.1f m' %(right_curverad), (33, 200), font, 1, (255, 255, 255), 2)

    # use the out_img_annotated for debugging, halve its size and put it top right
    half_width = original_img_overlaid.shape[1] // 2
    width = original_img_overlaid.shape[1]
    half_height = original_img_overlaid.shape[0] // 2
    original_img_overlaid[0:half_height, half_width:width, :] = \
        cv2.resize(out_img_annotated, (half_width, half_height))

    # before writing the image back into a video, convert back to RGB format
    if conv_rgb_to_bgr:
        original_img_overlaid = cv2.cvtColor(original_img_overlaid, cv2.COLOR_BGR2RGB)

    return original_img_overlaid


# Transformation source and destination
def transform_src_and_dst(img):
    height, width = img.shape[:2]
    # Origin is top left corner, y increases downwards
    # source below keeps a good margin to the horizon to get clear markings all the way
    src = np.float32([[578, 464], [708, 464], [258, 682], [1050, 682]])
    dst = np.float32([[470, 0], [width - 470, 0], [450, height], [width - 450, height]])
    return src, dst


# For each video, process each frame and output the resulting video
def process_video(input, output, process_image_fun):
    clip = VideoFileClip(input)
    # Uncomment to generate small quick range, range in seconds (start, end)
    # clip = clip.subclip(38, 41)
    vid_clip = clip.fl_image(process_image_fun)
    vid_clip.write_videofile(output, audio=False)
    return


# Main, process the 3 videos
def lanelines_main():
    dist, mtx = calibration_params()
    part_process_image = partial(process_image, mtx=mtx, dist=dist)
    process_video('project_video.mp4', 'output_images/project_video_out.mp4', part_process_image)

    part_process_image = partial(process_image, mtx=mtx, dist=dist)
    process_video('challenge_video.mp4', 'output_images/challenge_video_out.mp4', part_process_image)

    part_process_image = partial(process_image, mtx=mtx, dist=dist)
    process_video('harder_challenge_video.mp4', 'output_images/harder_challenge_video_out.mp4', part_process_image)


if __name__ == "__main__":
    lanelines_main()

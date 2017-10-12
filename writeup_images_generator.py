import cv2
import numpy as np
import lanelines as ll


# Inputs
calibration_f_name = 'camera_cal/calibration1.jpg'
test1_f_name = 'test_images/test1.jpg'
straight_lines1_f_name = 'test_images/straight_lines1.jpg'
curve_f_name = 'test_images/test5.jpg'

# Outputs, should all go into output_images directory
undistorted_calibration_f_name = 'output_images/undistorted_calibration.jpg'
undistorted_test1_f_name = 'output_images/test1_undistorted.jpg'
color_binary_f_name = 'output_images/test1_thresholds.jpg'
warped_f_name = 'output_images/straight_lines1_warped.jpg'
curve_warped_annotated_f_name = 'output_images/curve_warped_annotated.jpg'

def main():
    # Create undistorted version of distorted calibration image and test_images/test1.jpg
    dist, mtx = ll.calibration_params()
    calibration_image = cv2.imread(calibration_f_name)
    undistort_image = ll.undistort_image(calibration_image, mtx, dist)
    cv2.imwrite(undistorted_calibration_f_name, undistort_image)

    test1_img = cv2.imread(test1_f_name)
    undistorted_test1_image = ll.undistort_image(test1_img, mtx, dist)
    cv2.imwrite(undistorted_test1_f_name, undistorted_test1_image)


    # Generate combined thresholds of the previous image
    color_binary, combined_binary = ll.threshold_pipeline(undistorted_test1_image)
    color_binary = np.multiply(255, color_binary).astype(np.int)
    cv2.imwrite(color_binary_f_name, color_binary)

    # Generate warped image, the straight_lines1.jpg since that one should have vertical lane lines
    straight_lines1_img = cv2.imread(straight_lines1_f_name)
    straight_lines1_img = ll.undistort_image(straight_lines1_img, mtx, dist)
    # OK, not ideal to copy code, but easiest way...
    height, width = straight_lines1_img.shape[:2]
    src = np.float32([[610, 441], [669, 441], [258, 682], [1049, 682]])
    dst = np.float32([[450, 0], [width - 450, 0], [450, height], [width-450, height]])
    M, warped_img = ll.dashboard_to_overhead(straight_lines1_img, src, dst)
    cv2.imwrite(warped_f_name, warped_img)

    # Generate warped image with annotated lane lines
    curve_image = cv2.imread(curve_f_name)
    curve_annotated_image = process_image_to_annotaded_overhead(curve_image, mtx, dist)
    cv2.imwrite(curve_warped_annotated_f_name, curve_annotated_image)



    #cv2.imshow("bluegreen-thresholds", color_binary)
    #cv2.imshow("combined-thresholds", combined_binary)
    #cv2.imshow("undistorted_test1_image", undistorted_test1_image)
    #cv2.waitKey(20000)

    return


def process_image_to_annotaded_overhead(original_img, mtx, dist):
    undistorted_img = ll.undistort_image(original_img, mtx, dist)
    color_binary, combined_binary = ll.threshold_pipeline(undistorted_img)
    height, width = undistorted_img.shape[:2]
    src = np.float32([[610, 441], [669, 441], [258, 682], [1049, 682]])
    dst = np.float32([[450, 0], [width - 450, 0], [450, height], [width-450, height]])
    M, binary_warped = ll.dashboard_to_overhead(combined_binary, src, dst)
    # Mark left/right lines with colors, calculate left/right fit polynomials
    out_img, ploty, left_fitx, right_fitx, left_lane_center, right_lane_center = \
        ll.find_lane_lines(binary_warped)
    return out_img


main()

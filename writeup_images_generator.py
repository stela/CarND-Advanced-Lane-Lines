import cv2
import numpy as np
import lanelines as ll


# Inputs
calibration_f_name = 'camera_cal/calibration1.jpg'
test1_f_name = 'test_images/test1.jpg'

# Outputs, should all go into output_images directory
undistorted_calibration_f_name = 'output_images/undistorted_calibration.jpg'
undistorted_test1_f_name = 'output_images/test1_undistorted.jpg'
color_binary_f_name = 'output_images/test1_thresholds.jpg'

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

    #cv2.imshow("bluegreen-thresholds", color_binary)
    #cv2.imshow("combined-thresholds", combined_binary)
    #cv2.imshow("undistorted_test1_image", undistorted_test1_image)
    #cv2.waitKey(20000)

    return



main()

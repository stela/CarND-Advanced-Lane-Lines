import cv2
import lanelines as ll

# Inputs
calibration_f_name = 'camera_cal/calibration1.jpg'

# Outputs, should all go into output_images directory
undistorted_calibration_f_name = 'output_images/undistorted_calibration.jpg'

def main():
    # Create undistorted version of distorted calibration image
    dist, mtx = ll.calibration_params()
    calibration_image = cv2.imread(calibration_f_name)
    undistort_image = ll.undistort_image(calibration_image, mtx, dist)
    cv2.imwrite(undistorted_calibration_f_name, undistort_image)

    

    return



main()

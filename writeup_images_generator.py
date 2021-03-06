import cv2
import numpy as np
import lanelines as ll


# Inputs
calibration_f_name = 'camera_cal/calibration1.jpg'
test1_f_name = 'test_images/test1.jpg'
test2_f_name = 'test_images/test2.jpg'
straight_lines1_f_name = 'test_images/straight_lines1.jpg'
curve_f_name = 'test_images/test5.jpg'
dirty_f_name = 'test_images/dirty_border_noise.png'
few_lane_pixels_f_name = 'test_images/few_lane_pixels.png'

# Outputs, should all go into output_images directory
undistorted_calibration_f_name = 'output_images/undistorted_calibration.jpg'
undistorted_test1_f_name = 'output_images/test1_undistorted.jpg'
color_binary_f_name = 'output_images/test1_thresholds.jpg'
color_binary_curve_f_name = 'output_images/curve_thresholds.jpg'
color_binary_dirty_f_name = 'output_images/dirty_thresholds.jpg'
warped_f_name = 'output_images/straight_lines1_warped.jpg'
warped_dirty_f_name = 'output_images/dirty_warped.jpg'
curve_warped_annotated_f_name = 'output_images/curve_warped_annotated.jpg'
dirty_warped_annotated_f_name = 'output_images/dirty_border_noise_annotated.jpg'
test2_annotated_f_name = 'output_images/test2_annotated.jpg'
test2_result_f_name = 'output_images/test2_result.jpg'
curve_result_f_name = 'output_images/curve_result.jpg'
dirty_result_f_name = 'output_images/dirty_border_noise_result.jpg'
few_lane_pixels_annotated_f_name = 'output_images/few_lane_pixels_annotated.jpg'


def main():
    # Create undistorted version of distorted calibration image and test_images/test1.jpg
    dist, mtx = ll.calibration_params()
    calibration_image = cv2.imread(calibration_f_name)
    undistort_image = ll.undistort_image(calibration_image, mtx, dist)
    cv2.imwrite(undistorted_calibration_f_name, undistort_image)

    test1_img = cv2.imread(test1_f_name)
    undistorted_test1_image = ll.undistort_image(test1_img, mtx, dist)
    cv2.imwrite(undistorted_test1_f_name, undistorted_test1_image)

    curve_image = cv2.imread(curve_f_name)
    undistorted_curve_image = ll.undistort_image(curve_image, mtx, dist)

    dirty_image = cv2.imread(dirty_f_name)
    undistorted_dirty_image = ll.undistort_image(dirty_image, mtx, dist)

    straight_lines1_img = cv2.imread(straight_lines1_f_name)
    undistorted_straight_lines1_image = ll.undistort_image(straight_lines1_img, mtx, dist)

    few_lane_pixels_img = cv2.imread(few_lane_pixels_f_name)
    test2_img = cv2.imread(test2_f_name)

    # Generate combined thresholds of the previous images
    color_binary, combined_binary = ll.threshold_pipeline(undistorted_test1_image)
    color_binary = np.multiply(255, color_binary).astype(np.int)
    cv2.imwrite(color_binary_f_name, color_binary, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    color_binary, combined_binary = ll.threshold_pipeline(undistorted_curve_image)
    color_binary = np.multiply(255, color_binary).astype(np.int)
    cv2.imwrite(color_binary_curve_f_name, color_binary, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    color_binary, combined_binary = ll.threshold_pipeline(undistorted_dirty_image)
    color_binary = np.multiply(255, color_binary).astype(np.int)
    cv2.imwrite(color_binary_dirty_f_name, color_binary, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate warped image, the straight_lines1.jpg since that one should have vertical lane lines
    src, dst = ll.transform_src_and_dst(undistorted_straight_lines1_image)
    M, warped_img = ll.dashboard_to_overhead(undistorted_straight_lines1_image, src, dst)
    cv2.imwrite(warped_f_name, warped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate warped image, the dirty.png since that one should have vertical lane lines
    src, dst = ll.transform_src_and_dst(undistorted_dirty_image)
    M, warped_img = ll.dashboard_to_overhead(undistorted_dirty_image, src, dst)
    cv2.imwrite(warped_dirty_f_name, warped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate warped images with annotated lane lines
    curve_annotated_image = process_image_to_annotaded_overhead(curve_image, mtx, dist)
    cv2.imwrite(curve_warped_annotated_f_name, curve_annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    dirty_annotated_image = process_image_to_annotaded_overhead(dirty_image, mtx, dist)
    cv2.imwrite(dirty_warped_annotated_f_name, dirty_annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    few_lane_pixels_annotated_image = process_image_to_annotaded_overhead(few_lane_pixels_img, mtx, dist)
    cv2.imwrite(few_lane_pixels_annotated_f_name, few_lane_pixels_annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    test2_annotated_image = process_image_to_annotaded_overhead(few_lane_pixels_img, mtx, dist)
    cv2.imwrite(test2_annotated_f_name, test2_annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Generate final video frames using sample images
    curve_result_img = ll.process_image(curve_image, mtx, dist, False)
    cv2.imwrite(curve_result_f_name, curve_result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    dirty_result_img = ll.process_image(dirty_image, mtx, dist, False)
    cv2.imwrite(dirty_result_f_name, dirty_result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    test2_result_img = ll.process_image(test2_img, mtx, dist, False)
    cv2.imwrite(test2_result_f_name, test2_result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return


# Keep this method in sync with lanelines.process_image(), except for RGB/BGR mixup
def process_image_to_annotaded_overhead(original_img, mtx, dist, conv_rgb_to_bgr=False):
    if conv_rgb_to_bgr:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    undistorted_img = ll.undistort_image(original_img, mtx, dist)
    color_binary, combined_binary = ll.threshold_pipeline(undistorted_img)
    src, dst = ll.transform_src_and_dst(undistorted_img)
    M, binary_warped = ll.dashboard_to_overhead(combined_binary, src, dst)
    # Mark left/right lines with colors, calculate left/right fit polynomials
    out_img, lane_lines = \
        ll.find_lane_lines(binary_warped, None)

    if conv_rgb_to_bgr:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    return out_img


main()

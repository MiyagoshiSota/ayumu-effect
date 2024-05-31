import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import mediapipe as mp

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (0, 0, 0)  # white

# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(
    model_asset_path='./models/selfie_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# 画像の表示
filename = "./imgs/PSXgfeg9rZtrnvNut7D8uQ_Human_anatomy.jpg_.jpg"
imgCV = cv2.imread(filename)

#
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgCV)

    # Perform segmentation
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    fg_image = np.zeros(imgCV.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(imgCV.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(), ) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)

    # Display the segmented image
    cv2.imshow("image", output_image)
    cv2.waitKey(0)

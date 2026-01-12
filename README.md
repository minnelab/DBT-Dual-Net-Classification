Code for binary classification on BCS-DBT data: normal vs. abnormal / benign vs. malignant.


Pre-processing: preprocess 3D BCS-DBT data, including extracting ROI patches; generating tumor bounding-box annotations; sampling background data.

codes_classify_benign_malignant: the folder contains different processing scripts, categorized by dataset pre-processing.

--patch: the dataset uses patch-based samples with sizes 512×512, 224×224, or 256×256. Experiments show patch size has little impact on metrics.
vote: includes a voting mechanism.

--HSV: adjust in the HSV color space, then convert back to RGB with OpenCV. Input is a single grayscale image; after preprocessing, a pseudo-RGB 3-channel image is obtained.
whole: the dataset uses full images at size 1024×2048.

--3D_patch: the slice thickness in 3D is fixed and can be configured in the preprocessing script. Assuming the tumor is spherical or cubic, slice thickness can be computed from the CSV-provided width and height. Values vary; the minimum slice thickness is about 10. Therefore, when extracting 3D patches, use 10 as the standard slice thickness.

--CAM_visualize: use trained .pth model files to inspect model attention regions.

--self_supervised: approach inspired by https://arxiv.org/pdf/2408.10600.

--probing_CLIP: combine CLIP with our own trained model..
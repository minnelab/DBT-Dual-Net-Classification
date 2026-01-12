import os
import pydicom
from pydicom.pixel_data_handlers import pylibjpeg_handler
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
import nibabel as nib
from tqdm import tqdm
import gc
from typing import List
from duke_dbt_data import dcmread_image, read_boxes, evaluate, draw_box
import cv2
import zipfile
import os
from typing import List

def normalize_image(image):
    image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)   
    return image_2d_scaled    

def normalize_volume(volume):
    p10 = np.percentile(volume, 0.1)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    # m = np.mean(volume, axis=(0, 1, 2))
    # s = np.std(volume, axis=(0, 1, 2))
    # volume = (volume - m) / s
    return volume


def get_patient_and_study_id(dicom_file_path):
    """
    Extract Patient ID and Study ID from a DICOM file.
    Parameters:
    - dicom_file_path: Path to the DICOM file.

    Returns:
    - patient_id: The Patient ID extracted from the DICOM file.
    - study_uid: The Study ID extracted from the DICOM file.
    """
    ds = pydicom.dcmread(dicom_file_path)    
    patient_id = ds.PatientID
    # study_uid = ds.StudyInstanceUID
    study_uid = ds.StudyID
    return patient_id, study_uid

def collect_dicom_files(root_dir: str) -> List[str]:
    """Collect all DICOM file paths from the directory structure."""
    dicom_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(subdir, file))
    return dicom_files

def save_slices_based_on_csv(csv_data, dicom_dir, save_dir,depth,surround_dis):
    # Load the CSV file

    # Create a DataFrame to save metadata
    # df = pd.DataFrame(columns=['PatientID', 'StudyUID', 'view', 'img_path', 'Normal', 'Actionable', 'Benign', 'Cancer'])
    # df = pd.DataFrame(columns=['PatientID', 'StudyUID', 'view', 'img_path', 'Class'])
    # df = pd.DataFrame(columns=['PatientID', 'StudyUID', 'view', ])
    os.makedirs(save_dir, exist_ok=True)
    count = 0 
    for (key, row) in tqdm(csv_data.iterrows(), total=len(csv_data)):
        print(key,row)
        # if key<200:
        #     continue
        view = row['View']
        patient_id =row["PatientID"]
        study_uid = row["StudyUID"]
        dicom_file = row["descriptive_path"]
        dicom_file_path = os.path.join(dicom_dir, dicom_file)
        #dicom_file_path = dicom_file_path.replace('000000-', '000000-NA-')
        volume = dcmread_image(dicom_file_path,view)
        # print("volume.shape",volume.shape, patient_id, study_uid, matching_rows["StudyUID"])                    
        x = row['X']
        y = row['Y']
        width = row['Width']
        height = row['Height']
        slice_idx = row['Slice']

        # x = row['X1']
        # y = row['Y1']
        # x2 = row['X2']
        # y2 = row['Y2']
        # height = y2 - y
        # width = x2 -x
        # print("row",key,row)
        # print("row['Normal'], row['Actionable'], row['Benign'], row['Cancer']",row['Normal'], row['Actionable'], row['Benign'], row['Cancer'])
        # Ensure the slice index is within bounds
        start_idx = slice_idx-depth
        end_idx = slice_idx+depth
        y_start = y-surround_dis
        y_end = y+surround_dis+height
        x_start = x-surround_dis
        x_end = x+surround_dis+width
        # if y_start < 0:
        #     y_start = 0
        # if x_end > volume.shape[1]:
        #     x_end = volume.shape[1]
        # if x_start < 0:
        #     x_start = 0
        # if y_end > volume.shape[2]:
        #     y_end = volume.shape[2]
        # slice_volume = volume[start_idx:end_idx, x_start:x_end ,y_start:y_end]

        if  start_idx < 0:
            start_idx = 0
        if end_idx > volume.shape[0]:
            end_idx = volume.shape[0]

        if y_start < 0:
            y_start = 0
        if y_end > volume.shape[1]:
            y_end = volume.shape[1]
        if x_start < 0:
            x_start = 0
        if x_end > volume.shape[2]:
            x_end = volume.shape[2]
        img = normalize_volume(normalize_image(volume[slice_idx]))
        img = draw_box(img, x, y, width, height)
        cv2.imwrite(os.path.join(save_dir, f"{patient_id}_{study_uid}_{view}_slice_{slice_idx}.png"), img)

        slice_volume = volume[start_idx:end_idx, y_start:y_end ,x_start:x_end]
        slice_name = f"{patient_id}_{study_uid}_{view}_slice_{slice_idx}.nii.gz"
        slice_nifti = nib.Nifti1Image(slice_volume, np.eye(4))
        print(start_idx,end_idx, y_start,y_end ,x_start,x_end,volume.shape,slice_name)
        
        nib.save(slice_nifti, os.path.join(save_dir, slice_name))
        count = count + 1
        # df = pd.concat([df, pd.DataFrame([patient_id, study_uid, view, os.path.join(save_dir, slice_name),row['Class'] ])], ignore_index=True)
        # df = pd.concat([df, pd.DataFrame([patient_id, study_uid, view, os.path.join(save_dir, slice_name),
        #                 row['Normal'], row['Actionable'], row['Benign'], row['Cancer'] ])], ignore_index=True)
        # df.append( [patient_id, study_uid, view, os.path.join(save_dir, slice_name),
        #             row['Normal'], row['Actionable'], row['Benign'], row['Cancer'] ] )
        gc.collect()
    csv_data.to_csv(os.path.join(save_dir, 'slices_metadata.csv'), index=False)


# # Define file paths
# boxes_csv_path = '/content/Breast-Cancer-Screening-DBT/boxes.csv'
# filepaths_csv_path = '/content/Breast-Cancer-Screening-DBT/filepaths.csv'
# labels_csv_path = '/content/Breast-Cancer-Screening-DBT/labels.csv'
# predictions_csv_path = '/content/Breast-Cancer-Screening-DBT/predictions.csv'
# # Process the DICOM files
# process_dicom_files(dicom_files, boxes_csv_path, filepaths_csv_path, labels_csv_path, predictions_csv_path)



depth  = 5
surround_dis = 10
#boxes = "/proj/berzelius-2023-99/users/x_zyang/train_phase2/duke_label/boxes-train-v2.csv"
#train = "/proj/berzelius-2023-99/users/x_zyang/train_phase2/duke_label/train-v2.csv"


# 我的数据
boxes = "F:/Dataset/Train_new/manifest-1617905855234/DBT-labels/BCS-DBT-boxes-train-v2.csv"
train = "F:/Dataset/Train_new/manifest-1617905855234/DBT-labels/BCS-DBT-file-paths-train-v2.csv"


df_box_data = pd.read_csv(boxes)
df_train_data = pd.read_csv(train)
merge_df = pd.merge(df_box_data, df_train_data,"left",["PatientID","StudyUID","View"])
# csv_path = "/proj/berzelius-2023-99/users/x_zyang/train_phase2/DBT_phase2_train_bboxes.csv"
#dicom_dir = '/proj/berzelius-2023-99/users/x_zyang/dukeDBTdata/'
#我的数据

dicom_dir = "F:/Dataset/Train_new/manifest-1617905855234"
#save_dir = f'/proj/berzelius-2023-99/users/x_zyang/dukeDBTdata/3d_slices_nii_3D_{surround_dis}_{depth}P/'

#我的数据
save_dir = f'F:/Dataset/Train_new/manifest-1617905855234/Breast-Cancer-Screening-DBT/3d_slices_nii_3D_{surround_dis}_{depth}P/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Collect all DICOM file paths
dicom_files = collect_dicom_files(dicom_dir)

# Process the DICOM files based on the CSV data
save_slices_based_on_csv(merge_df, dicom_dir, save_dir, depth, surround_dis)

print("******************************************")
print(f"Found {len(dicom_files)} DICOM files.")
print(dicom_files[0])
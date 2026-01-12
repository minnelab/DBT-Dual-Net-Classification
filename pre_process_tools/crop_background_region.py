import os, gc
import random
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from duke_dbt_data import dcmread_image, read_boxes, evaluate, draw_box
import csv

def random_crop_except_bbox(image, bbox, crop_size):
    """
    Randomly crop an image except for the bounding box area.
    Args:
    - image (PIL.Image): The input image.
    - bbox (tuple): Bounding box in the format (x_min, y_min, width, height).
    - crop_size (tuple): Desired crop size (width, height).

    Returns:
    - cropped_image (PIL.Image): The cropped image.
    """
    img_width, img_height = image.size
    crop_width, crop_height = crop_size
    x_min, y_min, box_width, box_height = bbox
    x_max = x_min + box_width
    y_max = y_min + box_height

    # Ensure the crop size is smaller than the image size
    if crop_width > img_width or crop_height > img_height:
        return False
        raise ValueError("Crop size must be smaller than the image dimensions.")
    # Define the cropable regions (outside the bounding box)
    x_candidates = list(range(0, int(x_min - crop_width + 1))) + list(range(int(x_max), int(img_width - crop_width + 1)))
    y_candidates = list(range(0, int(y_min - crop_height + 1))) + list(range(int(y_max), int(img_height - crop_height + 1)))
    # print("y_candidates",y_candidates,"x_candidates",x_candidates)
    if not x_candidates or not y_candidates:
        raise ValueError("Crop size is too large, or the bounding box takes up too much space.")
    # Choose random top-left crop corner from the valid regions
    x_crop = random.choice(x_candidates)
    y_crop = random.choice(y_candidates)
    # Perform the crop
    cropped_image = image.crop((x_crop, y_crop, x_crop + crop_width, y_crop + crop_height))
    return cropped_image

# Function to check if more than 50% of the crop is black
def is_black_background(cropped_image, black_threshold=10, black_percentage_threshold=90):
    """
    Check if more than a certain percentage of the image is black.
    Args:
    - cropped_image (PIL.Image): The cropped image.
    - black_threshold (int): The intensity value below which a pixel is considered black.
    - black_percentage_threshold (float): The percentage threshold for considering the image as having too much black.

    Returns:
    - is_black (bool): True if more than the threshold percentage of the image is black.
    """
    # Convert image to numpy array
    image_np = np.array(cropped_image)
    # Check if the image is grayscale or RGB
    if len(image_np.shape) == 2:  # Grayscale
        mask_black = image_np < black_threshold
    else:  # RGB
        mask_black = np.all(image_np < black_threshold, axis=-1)
    # Calculate the percentage of black pixels
    black_percentage = np.sum(mask_black) / mask_black.size * 100
    return black_percentage > black_percentage_threshold

# Example usage
if __name__ == "__main__":
    #data_path = "/proj/berzelius-2023-99/users/x_zyang/dukeDBTdata/all_data"
    data_path = "F:/Dataset/Test/manifest-1617905855234"
    #base_path = '/proj/berzelius-2023-99/users/x_zyang/dukeDBTdata/duke_label'
    base_path = "F:/Dataset/Test/manifest-1617905855234/DBT-labels"
    output_path = 'F:/Dataset/Test/patch'
    #output_path = '/proj/berzelius-2023-99/users/x_zyang/train_phase2/patch_all/all_patch_grey'


    df = read_boxes(boxes_fp=base_path+"/BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv", filepaths_fp=base_path+"/BCS-DBT-file-paths-test-v2.csv") 
    label_csv = 'patch_label.csv'
    box_table = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv"))
    label_list = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-labels-test-PHASE-2.csv"))
    train_list = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-file-paths-test-v2.csv"))
    merge_df = pd.merge(label_list, train_list,"left",["PatientID","StudyUID","View"])
    merge_df = pd.merge(merge_df, box_table,"left",["PatientID","StudyUID","View"])
    print("merge_df",len(merge_df))
    image_csv= []
    for (key, row) in merge_df.iterrows():
        if row["descriptive_path"] is np.nan:
            continue
        view = row["View"]
        PatientID = row["PatientID"]
        StudyUID = row["StudyUID"]
        # print('row["descriptive_path"]', row["descriptive_path"],row["PatientID"],row["StudyUID"], row["Normal"])
        if row["Normal"]==1:
            image_path = os.path.join(data_path, row["descriptive_path"])
            
            image_path = image_path.replace('000000-', '000000-NA-')
            if os.path.exists(image_path):
                image3D = dcmread_image(fp=image_path, view=view)
            else:
                continue
            
            VolumeSlices = image3D.shape[0]
            for slice_index in range(VolumeSlices):
                for time in range(3):
                    crop_width = random.randint(100, 500)
                    crop_height = random.randint(100, 500)                
                    # print("image3D.shape",image3D.shape)                
                    bbox = (0, 0, 0, 0)  # Modify as per your bounding box
                    crop_size = (crop_width, crop_height)  # Modify as per your desired crop size
                    # Perform the random crop except for the bounding box area
                    image = image3D[slice_index]
                    image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0
                    image_2d_scaled = np.uint8(image_2d_scaled)
                    image = Image.fromarray(image_2d_scaled)
                    filename =PatientID+"_"+StudyUID+"_"+view+"_"+str(slice_index)+"_"+str(time)+".png"
                    cropped_image = random_crop_except_bbox(image, bbox, crop_size)
                    # Check if the cropped image has more than 50% black background
                    if is_black_background(cropped_image):
                        print("The cropped image has too much black background.")
                    else:
                        image_csv.append( [PatientID, StudyUID, filename, view, 0, 0, crop_width, crop_height, slice_index, VolumeSlices, 0])
                        cropped_image.save(os.path.join(output_path, filename))
                        print("Cropped image saved successfully!")
                        # cv2.imwrite(os.path.join(output_path, filename), cropped_image)

        if row["Benign"]==1 or row["Cancer"]==1:
            image_path = os.path.join("F:/Dataset/Test/manifest-1617905855234", row["descriptive_path"])
            image_path = image_path.replace('000000-', '000000-NA-')
            image3D = dcmread_image(fp=image_path, view=view)
            VolumeSlices = image3D.shape[0]
            for slice_index in range(VolumeSlices):
                for time in range(3):
                    bbox = (row["X"], row["Y"], row["Width"], row["Width"])  # Modify as per your bounding box
                    crop_width = random.randint(100, 500)
                    crop_height = random.randint(100, 500)
                    crop_size = (crop_width, crop_height)
                    image = image3D[slice_index]
                    image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0
                    image_2d_scaled = np.uint8(image_2d_scaled)
                    image = Image.fromarray(image_2d_scaled)
                    filename =PatientID+"_"+StudyUID+"_"+view+"_"+str(slice_index)+"_"+str(time)+".png"
                    cropped_image = random_crop_except_bbox(image, bbox, crop_size)
                    if is_black_background(cropped_image):
                        print("The cropped image has too much black background.")
                    else:
                        image_csv.append( [PatientID, StudyUID, filename, view, 0, 0, crop_width, crop_height, slice_index, VolumeSlices, 0])                        
                        cropped_image.save(os.path.join(output_path, filename))
                        print("Cropped image saved successfully!")
                        # cv2.imwrite(os.path.join(output_path, filename), cropped_image)
        # del image3D
        print("image_path",image_path)
        gc.collect()

    with open(os.path.join(base_path, label_csv), 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PatientID", "StudyUID", "Filename", "View",  "X", "Y", "Width", "Height" , "Slice", "VolumeSlices","Class"])
        
    with open(os.path.join(base_path, label_csv), 'a+', newline='') as file:
        writer = csv.writer(file)
        for (PatientID, StudyUID, filename, view,x,y,w,h,slice_index,VolumeSlices,_) in image_csv:
            writer.writerow([PatientID, StudyUID, filename, view,  x,y,w,h,slice_index,VolumeSlices,0])

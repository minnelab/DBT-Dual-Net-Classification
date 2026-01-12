import os, gc
import matplotlib.pyplot as plt
from duke_dbt_data import dcmread_image, read_boxes, draw_box
import omidb
import numpy as np
import pandas as pd
import cv2, csv, time
import json

#base_path = '/proj/berzelius-2023-99/users/x_zyang/dukeDBTdata/duke_label'
#base_path = "F:/Dataset/Train_normal_more_cases/manifest-1617905855234/DBT-labels"
base_path = "F:/Dataset/Val/manifest-1617905855234/DBT-val_list"
#stage = "train"
stage = "val"
generate_all = False

# train
if stage=="train":
    df = read_boxes(boxes_fp=base_path+"/BCS-DBT-boxes-train-v2.csv", filepaths_fp=base_path+"/BCS-DBT-file-paths-train-v2.csv")
    box_table = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-boxes-train-v2.csv"))
    label_lists = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-labels-train-v2.csv"))
    train_list = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-file-paths-train-v2.csv"))
# validation
if stage=="val":
    df = read_boxes(boxes_fp=base_path+"/BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv", filepaths_fp=base_path+"/BCS-DBT-file-paths-validation-v2.csv")
    box_table = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv"))
    label_lists = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv"))
    train_list = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-file-paths-validation-v2.csv"))
# test
if stage=="test":
    df = read_boxes(boxes_fp=base_path+"/BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv", filepaths_fp=base_path+"/BCS-DBT-file-paths-test-v2.csv")
    box_table = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv"))
    label_lists = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-labels-test-PHASE-2.csv"))
    train_list = pd.DataFrame(pd.read_csv(base_path + "/BCS-DBT-file-paths-test-v2.csv"))

merge_df = pd.merge(df, label_lists,"left",["PatientID","StudyUID","View"])
extra_size = 5
debug = False
rgb = True

# 2.5d image offset
offset = 3

#output_path = "F:/Dataset/Train_normal_more_cases/train_phase2/"
output_path = "F:/Dataset/Val/val_phase2/"

#filtered_raw_data = "F:/Dataset/Train_normal_more_cases/manifest-1617905855234/"
filtered_raw_data = "F:/Dataset/Val/manifest-1617905855234/"
output_folder = 'rgb_all_png'

folder_name = stage+ "_extend_DBT_slice_rgb"
whole_image_fold = folder_name+ str(offset)
patch_image_fold = folder_name + "_patch" + str(offset) #rgb_patch

all_whole_image_fold =  stage+ "allBMDBTSliceOriRGB" +str(offset)
label_csv_rgb = stage+"_extend_rgb_DBT_phase2_train_bboxes_"+str(offset)+".csv"

label_csv_rgb_path = os.path.join(output_path, output_folder, label_csv_rgb)
print("label_csv_rgb_path",label_csv_rgb_path)
if not os.path.exists(os.path.join(output_path,output_folder,whole_image_fold)):
    os.mkdir(os.path.join(output_path,output_folder,whole_image_fold))
if not  os.path.exists(os.path.join(output_path,output_folder,patch_image_fold)):
    os.mkdir(os.path.join(output_path,output_folder,patch_image_fold))


def get_normal_BBox(image,bbox):
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape,dtype=np.uint8)
    img2[output == max_label] = 255
    contours, hierarchy = cv2.findContours(img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    aux_im = img2
    x,y,w,h = cv2.boundingRect(cnt)
    if (debug):
        cv2.rectangle(aux_im,(x,y),(x+w,y+h),(255,0,0),5)
        plt.imshow(aux_im)
        plt.show()
    out_bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)
    return out_bbox, img2                   # returns bounding box and mask image. 


def copy_case_crop(view, client, episode, image_rgb, side, bbox_roi, slice_index, filename, VolumeSlices, lesion_type):
    # print(filename,image_rgb.shape)
    comp = image_rgb.shape[0]
    for nc in range(comp):
        image =  image_rgb[nc]
        dims = image.shape
        image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        if (side=='r' or side=='R'): # flip image and ROI coordinates.
            image_2d_scaled =cv2.flip(image_2d_scaled, 1)
            aux = bbox_roi.x2 
            bbox_roi.x2 = dims[1]-bbox_roi.x1 
            bbox_roi.x1 = dims[1]-aux
        dims = image_2d_scaled.shape
        # print("dimension",dims)
        if (nc == 0) : # same bbox for all slices
            bbox, mask = get_normal_BBox(image_2d_scaled,bbox_roi)
            print(bbox)
        image_crop = image_2d_scaled[bbox.y1:bbox.y2,bbox.x1:bbox.x2]
        # print(image_crop.shape)
        if (nc==0) and (comp>1): 
            out_rgb = np.zeros([image_crop.shape[0],image_crop.shape[1],comp])
        out_rgb[:,:,nc] = image_crop
    # choose the crop region
    aux_folder_patch = os.path.join(output_path,output_folder,patch_image_fold, filename)
    y1 = np.maximum(bbox_roi.y1-extra_size,0)
    y2 = np.minimum(bbox_roi.y2+extra_size,dims[0])
    x1 = np.maximum(bbox_roi.x1-extra_size,0)
    x2 = np.minimum(bbox_roi.x2+extra_size,dims[1])
    # image_crop = out_rgb[y1:y2,x1:x2]
    # print("image_crop",image_crop.shape)
    # cv2.imwrite(aux_folder_patch,image_crop)
    aux_folder = os.path.join(output_path,output_folder,whole_image_fold, filename)
    aux_folder_crop = os.path.join(output_path,output_folder,patch_image_fold, filename)
    print(aux_folder,out_rgb.shape, image_rgb.shape)
    w,h,d = out_rgb.shape
    cv2.imwrite(aux_folder,out_rgb)
    # adapt bbox_ROI to new cropped image (remove bbox.x1 and bbox.y1)
    bbox_roi.x1 = bbox_roi.x1 - bbox.x1
    bbox_roi.y1 = bbox_roi.y1 - bbox.y1
    bbox_roi.x2 = bbox_roi.x2 - bbox.x1
    bbox_roi.y2 = bbox_roi.y2 - bbox.y1
    if (side=='r' or side=='R'):
        image_rgb = np.flip(image_rgb, axis=2)
    # print("bbox_roi",bbox_roi, bbox,x1,x2,y1,y2,image_rgb.shape)
    crop_image = image_rgb[:,y1:y2,x1:x2]
    # print("crop_image",crop_image.shape)
    crop_image = np.transpose(crop_image,(1,2,0))
    cv2.imwrite(aux_folder_crop,crop_image)
    with open(label_csv_rgb_path, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([client, episode, filename, view, bbox_roi.x1, bbox_roi.y1, bbox_roi.x2, bbox_roi.y2, \
                slice_index, VolumeSlices, lesion_type, w, h])
        

def copy_case_raw(view, client, episode,image_rgb, side, bbox_roi, slice_index,filename):
    comp = image_rgb.shape[0]
    print(filename,image_rgb.shape,comp)
    out_rgb = np.zeros([image_rgb.shape[1],image_rgb.shape[2],comp])
    for nc in range(comp):
        image =  image_rgb[nc]
        dims = image.shape
        image_2d_scaled = (np.maximum(image,0) / image.max()) * 255.0
        image_2d_scaled = np.uint8(image_2d_scaled)
        out_rgb[:,:,nc] = image_2d_scaled
        # dims = image_2d_scaled.shape
        # print("dimension",dims)
        # if (nc == 0) : # same bbox for all slices
        #     bbox, mask = get_normal_BBox(image_2d_scaled,bbox_roi)
        # image_crop = image_2d_scaled[bbox.y1:bbox.y2,bbox.x1:bbox.x2]
        # if (nc==0) and (comp>1): 
        #     out_rgb = np.zeros([image_crop.shape[0],image_crop.shape[1],comp])
        # out_rgb[:,:,nc] = image_crop        
    # choose the crop region
    # aux_folder = output_path+output_folder+"/"+patch_image_fold+"/patch_"+ filename
    # aux_folder = os.path.join(output_path,output_folder,whole_image_fold,"patch_"+ filename)
    # aux_folder = output_path+output_folder+"/"+patch_image_fold+  "/" + filename
    aux_folder_whole = os.path.join(output_path,output_folder,all_whole_image_fold, filename)
    print("aux_folder_whole",aux_folder_whole)
    # aux_folder_patch = os.path.join(output_path,output_folder,whole_image_fold, "patch_"+  filename)
    cv2.imwrite(aux_folder_whole,out_rgb)


def convert_csv_to_json(name):
    i = 0 
    dataset_dicts=[]
    label_list = pd.DataFrame(pd.read_csv(name) )
    # print("label_list",label_list[:5])
    for id in range(len(label_list)):
        record = {}
        ann = []
        filename = label_list["Filename"][id]
        # if not os.path.exists(img_path):
        #     continue
        # get box
        col1 = box_table['StudyUID'] ==  label_list['StudyUID'][id]
        col2 = box_table['View'] ==  label_list['View'][id]
        col_final = box_table[col1&col2]
        record["file_name"] = filename
        record["image_id"] = int(i)
        record["height"] = int(label_list["Y2"][id])
        record["width"] = int(label_list["X2"][id])
        print("col_final",col_final)
        for index, line in col_final.iterrows():
            print("line",line)
            X1 = line['X1']
            Y1 = line['Y1']
            X2 = line['X2']
            Y2 = line['Y2']
            Width = X2 - X1
            side = line['View'][0]
            # if side == 'r':
            #     x = image.shape[1]-1-x-Width
            box = [int(X1),int(Y1),int(X2), int(Y2)]
            if box_table['Class'][0]=="c":
                obj = {'bbox':box, 
                        "bbox_mode": 0,
                        "segmentation":[],
                        "category_id": 1}
            else:
                obj = {'bbox':box, 
                        "bbox_mode": 0,
                        "segmentation":[],
                        "category_id": 0}                
            ann.append(obj)
        i +=1
        record["annotations"] = ann
        dataset_dicts.append(record)
    print("i",i)
    new_dict = {}
    for i in dataset_dicts:
        new_dict[i["file_name"]] = i
    name = os.path.basename(name)
    name = name.split(".")[0]
    with open(os.path.join(output_path, output_folder, name+".json"), "w") as outfile: 
        json.dump(new_dict, outfile)


def csv_to_json(csv_file, json_file):
    df = pd.DataFrame(pd.read_csv(csv_file))
    categories = [{'id': 1, 'name': 'benign'}, {'id': 2, 'name': 'malignant'}]
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 1
    annotation_id = 1

    # Iterate over rows in the CSV file
    for index, row in df.iterrows():
        # Assuming CSV columns are: image_id, file_name, width, height, category_id, x_min, y_min, x_max, y_max
        image_info = {
            "id": image_id,
            "file_name": row['Filename'],
            "width": row['width'],
            "height": row['height']
        }
        coco_data["images"].append(image_info)
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [row['X1'], row['Y1'], row['X2'] - row['X1'], row['Y2'] - row['Y1']],
            "area": (row['X2'] - row['X1']) * (row['Y2'] - row['Y1']),
            "iscrowd": 0
        }
        coco_data["annotations"].append(annotation)
        image_id += 1
        annotation_id += 1

    with open(json_file, "w") as json_file:
        json.dump(coco_data, json_file)


if __name__=="__main__":

    print(len(df))
    print(df[:5])
    for index,box_series in df.iterrows():
        view = box_series["View"]
        if stage!="train":
            print(f'box_series["descriptive_path"]',box_series["descriptive_path"])
            name = box_series["descriptive_path"].split("/")[3]
            new_name = name.split("-")[0] + "-NA-" + name.split("-")[1]

            file_path = os.path.join("/".join(box_series["descriptive_path"].split("/")[:3]),new_name,"1-1.dcm")
        else:
            file_path = box_series["descriptive_path"]
        image_path = os.path.join(filtered_raw_data, file_path)
        image3D = dcmread_image(fp=image_path, view=view)
        num_z = image3D.shape[0]
        side = view[0]
        x, y, w, h = box_series[["X", "Y", "Width", "Height"]]
        min_len = min(w,h)
        slice_len = min_len // 12
        offset_len = slice_len // 2
        bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)
        PatientID = box_series["PatientID"]
        StudyUID = box_series["StudyUID"]
        annotated_slice = box_series["Slice"]
        VolumeSlices = box_series["VolumeSlices"]
        lesion_type = box_series["Class"]
        if generate_all:
            # slice_index = box_series["Slice"]
            for slice_index in range(offset,num_z-offset):

                image_rgb = np.array([image3D[slice_index-offset],image3D[slice_index],image3D[slice_index+offset]])
                filename = PatientID+"_"+StudyUID+"_"+view+"_"+str(slice_index)+".png"    

                copy_case_raw(view, PatientID, StudyUID, image_rgb, side, bbox, slice_index, filename)

        else:
            start_slice_index = max(offset, annotated_slice + offset- offset_len)
            end_slice_index = min(VolumeSlices-offset, annotated_slice - offset + offset_len)
            if start_slice_index<=end_slice_index: end_slice_index = start_slice_index+1
            print("start_slice_index",start_slice_index,"end_slice_index",end_slice_index,"VolumeSlices",VolumeSlices, "offset_len",offset_len)                
            for slice_index in range(start_slice_index, end_slice_index):
                x, y, w, h = box_series[["X", "Y", "Width", "Height"]]
                bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)
                image_rgb = np.array([image3D[slice_index-offset], image3D[slice_index], image3D[slice_index+offset]])
                filename = PatientID+"_"+StudyUID+"_"+view+"_"+str(slice_index)+".png"
                copy_case_crop(view,PatientID, StudyUID, image_rgb, side, bbox, slice_index, filename, VolumeSlices, lesion_type)

        del image3D
        gc.collect()
    
    print('**************running file ends here************')
        
    # df = pd.DataFrame(pd.read_csv(label_csv_rgb_path))
    # df.sort_values(by=['PatientID'], inplace=True, ascending=True)
    # df.reset_index(drop=True, inplace=True)

    # name = os.path.basename(label_csv_rgb_path)
    # name = name.split(".")[0]   + ".json"
    # label_json_path = os.path.join(output_path, output_folder, name)
    # csv_to_json(label_csv_rgb_path,label_json_path)
    # train_json_path = label_json_path.split(".")[0] + "_train.json"
    # val_json_path = label_json_path.split(".")[0] + "_val.json"
    # split_coco_json(label_json_path, train_json_path, val_json_path)
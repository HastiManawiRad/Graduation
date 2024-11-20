import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
import math
import pandas as pd
import ifcopenshell
import ifcopenshell.util.element

import torchvision
from torch import uint8
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import v2 as T2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib import patches
import xml.etree.cElementTree as ET
import time
import glob
import json
from PIL import Image, ImageFont, ImageOps
#import engine
#import tqdm
#from engine import train_one_epoch, evaluate

def read_image(image_path):
    return Image.open(image_path)


class OutletsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.root_path = dataset_path
        self.files = glob.glob(self.root_path+"/*.xml")
        self.transforms = transforms
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        xml_path = os.path.join(self.root_path, self.files[idx])
        root = ET.parse(xml_path).getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            name_element = obj.find("name")
            if name_element is None:
                print(f"Missing 'name'element in {xml_path}")
                continue

            class_name = name_element.text.lower()
            bndbox = obj.find("bndbox")
            if bndbox is None:
                print(f"Missing 'bndbox' element in {xml_path}")
                continue

            try:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
            except AttributeError as e:
                print(f"Error parsing 'bndbox' in {xml_path}: {e}")
                continue
               
            boxes.append([xmin, ymin, xmax, ymax])

            if class_name == "outlet":
                class_index = 1
            else:
                class_index = 0

            labels.append(class_index)

        if not boxes or not labels:
            print(f"No objects found in {xml_path}")
            return None
        
        boxes_torch = torch.tensor(boxes, dtype=torch.float32)
        labels_torch = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes_torch, "labels": labels_torch}

        img_path = os.path.join(self.root_path, root.find("filename").text)
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)

        return img, target

#Early stopping for training
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True

def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

train_subset = 0
test_subset = 0

class ExtractRGB(object):
    def __call__(self, img):
        if isinstance(img, Image.Image) and img.mode =="RGBA":
            img = img.convert("RGB")
        return img

img_transforms = transforms.Compose([
    #transforms.Resize((224, 224)),
    ExtractRGB(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.compose(transforms)

def custom_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def loadData():

    #create dataset & split
    dataset = OutletsDataset(r"PATH TO YOUR DATASET", img_transforms)

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    dataset_len = len(dataset)
    train_len = int(dataset_len*train_ratio)
    val_len = int(dataset_len*val_ratio)
    test_len = dataset_len - train_len - val_len

    train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

    print(len(train_subset))
    print(len(val_subset))
    print(len(test_subset))

    return train_subset, test_subset, val_subset

def AP(test_subset):
    test_loader = DataLoader(test_subset, batch_size=8, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    return test_loader


def train(train_subset, test_subset, val_subset, img_transforms):
    BATCH_SIZE = 8
    #train_dataset = ColumnsDataset()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    #test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    
    num_classes = 2
    model = custom_model(num_classes)

    images, targets = next(iter(train_loader))
    images = list(image for image in images)

    model_targets = []

    for target in targets:
        print(target)
        if isinstance(target, dict):
                model_targets.append({
                    "boxes": target["boxes"],
                    "labels": target["labels"]
                })
        else:
            print(f"skipping non-dict target: {target}")


    output = model(images, model_targets)
    print(type(output))
    print(output)

    #Define optimizer & epochs
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    print()
    num_classes = 2
    model = custom_model(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.0001, weight_decay= 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    model.to(device)

    for images, targets in train_loader:
        print(targets)
        break


    # Validation function
    def validate_one_epoch(model, val_loader, device):
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                print(f"targets: {targets}")

                model.train()
                loss_dict = model(images,targets)
                model.eval()
                
                print(f"Type loss_dict: {type(loss_dict)}")
                print(f"Content loss_dict: {loss_dict}")

                if isinstance(loss_dict, dict):
                # Sum all the loss values in the dictionary
                    losses = sum(loss for loss in loss_dict.values() if isinstance(loss, torch.Tensor))
                else:
                    raise ValueError("Unexpected type for model output, expected dict")
            
                epoch_val_loss += losses.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"average validation loss {avg_val_loss}")
        return avg_val_loss

    #training loop
    def train_one_epoch(model, optimizer, train_loader, device):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        for i, (images, targets) in enumerate(train_loader):
            
            batch_start_time = time.time()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time
            speed = 1/batch_time
            print(f"[Batch {i+1}] Loss: {loss_value:.3f}, Speed: {speed:.2f} batches/sec")

            if not math.isfinite(loss_value):
                print(f"Loss is { loss_value}, stopping training")
                print(loss_dict)
                break

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

    #train model for n epochs
    num_epochs = 25
    epoch_losses = [] 
    validation_losses = []
    best_loss = float('inf')


    for epoch in range(num_epochs):
        print("----------------- Training Epoch {} -----------------".format(epoch+1))
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        epoch_losses.append(avg_loss)

        #validation 
        print(f"Starting validation for epoch {epoch+1}")
        avg_val_loss = validate_one_epoch(model, val_loader, device)
        validation_losses.append(avg_val_loss)

        lr_scheduler.step(avg_val_loss)
    
        # early stopping
        early_stopping(avg_loss, avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", epoch)
            break
        
        print(f"average loss {avg_loss}")
        print("finished training")

        #save model with best parameters
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss
        }
        torch.save(checkpoint, "best-model-parameters.testcase2.pt")


    epochs_completed = len(epoch_losses)

    plt.plot(range(1, epochs_completed + 1), epoch_losses, label="Training loss")
    plt.plot(range(1, epochs_completed + 1), validation_losses, label="Validation loss", linestyle='--')

    plt.xticks(range(1, epochs_completed + 1))

    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.savefig("Training_validation_loss_testcase2.pdf")


    

    return model



def test(model, image_directory, csv_file_path):
    print("Start testing process")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = custom_model(num_classes)
    checkpoint = torch.load("best-model-parameters.testcase2.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    csv_data = pd.read_csv(csv_file_path, sep=';', dtype={'Placement': object})
    print(csv_data.columns)
    csv_data.columns = csv_data.columns.str.strip()

    img_transforms = transforms.Compose([
        ExtractRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    placement_results = ["No Results"] * len(csv_data)
    

    for index, row in csv_data.iterrows():
        socket_id = row['Socket']
        image_filename = f"Socket{socket_id}.jpeg"
        image_path = os.path.join(image_directory, image_filename)

        print(f"Processing {image_filename} for Socket {socket_id}...")

        if not os.path.exists(image_path):
            print(f"Image for Socket {socket_id} not found: {image_filename}")
            placement_results[index] = "Image Not Found"
            continue

        test_image = Image.open(image_path)
        test_image = ImageOps.exif_transpose(test_image)
        test_image = img_transforms(test_image)
        test_image = test_image[:3, ...].to(device)

        print(f"Looking for image at: {image_path}")
        with torch.no_grad():
            predictions = model([test_image])
            pred = predictions[0] if predictions else None

        if pred is None or (pred["scores"].sum() == 0):
            print(f"No detections for Socket {socket_id}")
            placement_results[index] = "No detections"
            continue  

        confidence_treshold = 0.85
        keep = pred["scores"] > confidence_treshold

        if keep.sum().item() == 0:
            print(f"No detections for Socket {socket_id}")
            placement_results[index] = "Image Not Found"
            continue   
    
        pred_boxes = pred["boxes"][keep].cpu()
        pred_labels = pred["labels"][keep].cpu()
        pred_scores = pred["scores"][keep].cpu()

        formatted_labels = [f"Socket {i+1}: {score:.3f}" for i, score in enumerate(pred_scores)]

        print("Bbox coordinates:")
        for i, (box, label) in enumerate(zip(pred_boxes, formatted_labels)):
            x_min, y_min, x_max, y_max = box
            print(f"{label} (x_min: {x_min:.2f}, y_min: {y_min:.2f}, x_max: {x_max:.2f}, y_max: {y_max:.2f})")

        def get_intersection(box_expected, box_actual):
            x_min1, y_min1, x_max1, y_max1 = box_expected
            x_min2, y_min2, x_max2, y_max2 = box_actual

            x_min_int = max(x_min1, x_min2)
            y_min_int = max(y_min1, y_min2)
            x_max_int = min(x_max1, x_max2)
            y_max_int = min(y_max1, y_max2)

            int_width = max(0, x_max_int - x_min_int)
            int_height = max(0, y_max_int - y_min_int)

            intersection_region = int_width * int_height
            return intersection_region
        
        def comparison_boxes(pred_boxes, expected_box):
            expected_region  = (expected_box[2] - expected_box[0]) * (expected_box[3] - expected_box[1])
            placement_status = "Outside Region"

            for i, box in enumerate(pred_boxes):
                intersection_region = get_intersection(box, expected_box)

                pred_region = (box[2] - box[0]) *(box[3] - box[1])

                if intersection_region == pred_region and intersection_region > 0:
                    print(f"Socket {i + 1} falls within expected region.")
                    placement_status = "Within Region"
                    break
                elif intersection_region > 0:
                    print(f"Socket {i + 1} falls partially within expected region.")
                    placement_status = "Partially Within Region"
                
            return placement_status


        test_image = test_image.cpu()
        test_image = (255.0 * (test_image - test_image.min()) / (test_image.max() - test_image.min())).to(torch.uint8)
        test_image = test_image[:3, ...]
        output_image = draw_bounding_boxes(test_image, pred_boxes, labels=formatted_labels, colors="red", width=3)

        rect_width = 517
        rect_height = 272

        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))

        height, width = output_image.shape[1], output_image.shape[2]
        plt.axhline(y=height/2, color='green', linestyle=':', linewidth=1.5)
        plt.axvline(x=width/2, color='green', linestyle=':', linewidth=1.5)

        rect_x = (width / 2) - (rect_width / 2)
        rect_y = (height / 2) - (rect_height / 2)
        rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1.5, edgecolor='green', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)

        rect_x_min = rect_x
        rect_y_min = rect_y
        rect_x_max = rect_x + rect_width
        rect_y_max = rect_y + rect_height
        print(f"Socket {socket_id} expected coordinates: (x_min: {rect_x_min:.2f}, y_min: {rect_y_min:.2f}, x_max: {rect_x_max:.2f}, y_max: {rect_y_max:.2f})")
        
        plt.show()

        expected_box = [rect_x_min, rect_y_min, rect_x_max, rect_y_max]
        #print(f"Expected bounding box for Socket {socket_id}: {expected_box}")
        placement_results[index] = comparison_boxes(pred_boxes, expected_box)

        #print(f"Length of placement results: {len(placement_results)}")
        #print(f"Length of CSV data: {len(csv_data)}")

        csv_data['Placement'] = placement_results
        csv_data.to_csv(csv_file_path, sep=';', index=False)
        print(f"Updated CSV file at {csv_file_path}")



def Average_Precision(model, test_loader, save_path="resultsAP.json"):
    print("Start AP calculation process")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = custom_model(num_classes)
    checkpoint = torch.load("best-model-parameters.testcase2.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    map_metric = MeanAveragePrecision(iou_thresholds=[0.5])

    for images, targets in test_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                outputs = model(images)
            
            batch_predictions = []
            batch_ground_truths = []

            for i, output in enumerate(outputs):
                batch_predictions.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu")
                })

                batch_ground_truths.append({
                    "boxes": targets[i]["boxes"].to("cpu"),
                    "labels": targets[i]["labels"].to("cpu")
                })

            map_metric.update(batch_predictions, batch_ground_truths)
        
            result = map_metric.compute()
            print(f"AP at IoU=0.5: {result['map_50'].item():.4f}")

            with open(save_path, "w") as f:
                result_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
                json.dump(result_serializable, f, indent=4)
            print(f"Results saved to {save_path}")


def IFC_attribute(csv_file_path, ifc_file_path, asbuilt_ifc_path):
    csv_data = pd.read_csv(csv_file_path, sep=";")
    ifc_file = ifcopenshell.open(ifc_file_path)

    for _, row in csv_data.iterrows():
        guid = row['GUID']
        placement_status = row['Placement']
        socket_number = row["Socket"]

        ifc_socket = ifc_file.by_guid(guid)

        if ifc_socket:
            print(f"Found Socket {socket_number} (GUID: {guid})")

            if ifc_socket.is_a("IfcBuildingElementProxy"):
                print(f"Processing socket with type: {ifc_socket.is_a()}")


            existing_property_set = None
            for rel in ifc_file.by_type("IfcRelDefinesByProperties"):
                if rel.RelatedObjects and ifc_socket in rel.RelatedObjects:
                    if rel.RelatingPropertyDefinition.Name == "As-Built Data":
                        existing_property_set = rel.RelatingPropertyDefinition
                        print(f"Found existing 'As-Built Data' property set: {existing_property_set.Name}")
                        break

            if not existing_property_set:
                existing_property_set = ifc_file.create_entity(
                    "IfcPropertySet",
                    GlobalId=ifcopenshell.guid.new(),
                    OwnerHistory=ifc_socket.OwnerHistory,
                    Name="As-Built Data",
                    Description="Properties related to as-built socket placement"
                )
                
                rel_defines = ifc_file.create_entity(
                    "IfcRelDefinesByProperties",
                    GlobalId=ifcopenshell.guid.new(),
                    OwnerHistory=ifc_socket.OwnerHistory,
                    RelatedObjects=[ifc_socket],
                    RelatingPropertyDefinition=existing_property_set
                )
                ifc_file.add(rel_defines)
                print(f"Created new 'BIM Data' property and relationship")

            property_value_asbuilt = ifc_file.create_entity(
                "IfcPropertySingleValue",
                Name="As-Built Placement",
                Description="Placement As-Built object with respect to the As-Planned placement",
                NominalValue=ifc_file.create_entity("IfcLabel", placement_status),
                Unit=None
            )

            if existing_property_set.HasProperties:
                existing_properties = existing_property_set.HasProperties
                existing_property_set.HasProperties = existing_properties + (property_value_asbuilt,)
            else:
                existing_property_set.HasProperties = (property_value_asbuilt,)

            print(f"Added 'As-Built Placement' with value '{placement_status}' to {existing_property_set.Name}.")

    ifc_file.write(asbuilt_ifc_path)
    print(f"As-Built IFC file saved at {asbuilt_ifc_path}")



image_directory = (r"PATH TO YOUR TESTING BATCH")
csv_file_path = (r"PATH TO YOUR CSV FILE")
ifc_file_path = (r"PATH TO YOUR IFC MODEL")
asbuilt_ifc_path = (r"PATH TO YOUR AS-BUILT IFC MODEL")


#HASH OUT WHATEVER MODULE YOU DO NOT WANT TO RUN HERE
model = custom_model
train_subset, test_subset, val_subset = loadData()
test_loader = AP(test_subset)
train(train_subset, test_subset, val_subset, img_transforms)
test(model, image_directory, csv_file_path)
Average_Precision(model, test_loader)
IFC_attribute(csv_file_path, ifc_file_path, asbuilt_ifc_path)










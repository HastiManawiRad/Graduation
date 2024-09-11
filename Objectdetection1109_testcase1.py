import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
import math

import torchvision
from torch import uint8
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import v2 as T2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from torchvision.datasets import ImageFolder
import xml.etree.cElementTree as ET
import time
import glob
from PIL import Image
#import engine
#import tqdm
#from engine import train_one_epoch, evaluate

def read_image(image_path):
    return Image.open(image_path)



class ColumnsDataset(torch.utils.data.Dataset):
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

            if class_name == "column":
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

img_transforms = transforms.Compose([
    #transforms.Resize((224, 224)),
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
    dataset = ColumnsDataset(r"C:\Users\HMd5\OneDrive - BVGO\School\Master\Afstuderen\OD\images", img_transforms)
    #dataset_test = ColumnsDataset(r"C:\Users\HMd5\OneDrive - BVGO\School\Master\Afstuderen\OD\AUG_images", img_transforms)
    #indices = torch.randperm(len(dataset)).tolist()

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    dataset_len = len(dataset)
    train_len = int(dataset_len*train_ratio)
    val_len = int(dataset_len*val_ratio)
    test_len = dataset_len - train_len - val_len

    train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

    print(len(train_subset))
    print(len(val_subset))
    print(len(test_subset))

    return train_subset, test_subset, val_subset

def train(train_subset, test_subset, val_subset, img_transforms):
    BATCH_SIZE = 8
    #train_dataset = ColumnsDataset()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
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

    #inference:
    model.eval()
    x = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]
    predictions = model(x)

    #Define optimizer & epochs
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    print()
    num_classes = 2
    model = custom_model(num_classes)
    #lr terug naar 0.001 zetten?
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

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            batch_time = time.time()
            speed = (i+1)/(batch_time-start_time)
            print("[%5d] loss: %.3f, speed: %.2f" %
                (i, loss_value, speed))
            
            if not math.isfinite(loss_value):
                print(f"Loss is { loss_value}, stopping training")
                print(loss_dict)
                break

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

    #train model for n epochs
    num_epochs = 15
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

    epochs_completed = len(epoch_losses)

    plt.plot(range(1, epochs_completed + 1), epoch_losses, label="Training loss")
    plt.plot(range(1, epochs_completed + 1), validation_losses, label="Validation loss", linestyle='--')

    # Update x-ticks based on the completed epochs
    plt.xticks(range(1, epochs_completed + 1))

    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.show()

    #save model with best parameters
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_loss
        }
        torch.save(checkpoint, "best-model-parameters.1009.test.pt")
    #torch.save(model.state_dict(), 'best-model-parameters.pt')

    return model

def test(model):
    #load model with best parameters
    #testing loop
    print("Start testing process")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = custom_model(num_classes)
    checkpoint = torch.load("best-model-parameters.1009.test.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    test_image_path = r"C:\Users\HMd5\OneDrive - BVGO\School\Master\Afstuderen\OD\AUG_images\image00385_aug_3.jpeg"
    test_image = Image.open(test_image_path)

    img_transforms = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])
    
    test_image = img_transforms(test_image)
    test_image = test_image[:3, ...].to(device)
    
    #test_image = read_image(test_image_path)  # This reads the image as a tensor
    #test_image = convert_image_dtype(test_image, torch.float32)
    #print(f"Image loaded. Type: {type(test_image)}, Shape: {test_image.shape}")

    #eval_transform = get_transform(train=False)
    #print(f"Image after transformation. Type: {type(test_image)}, Shape: {test_image.shape}")
    with torch.no_grad():
        predictions = model([test_image])
        pred = predictions[0]

    test_image = test_image.cpu()
    test_image = (255.0 * (test_image - test_image.min()) / (test_image.max() - test_image.min())).to(torch.uint8)
    test_image = test_image[:3, ...]
    pred_labels = [f"columns: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(test_image, pred_boxes, pred_labels, colors="red", width=3)

    confidence_treshold = 0.9
    pred_boxes = pred["boxes"][pred["scores"] > confidence_treshold].cpu()
    score = pred["scores"][pred["scores"] > confidence_treshold].cpu()
    pred_labels = pred["labels"][pred["scores"] > confidence_treshold].cpu()

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


model = custom_model
train_subset, test_subset, val_subset = loadData()
#train(train_subset, test_subset, val_subset, img_transforms)
test(model)





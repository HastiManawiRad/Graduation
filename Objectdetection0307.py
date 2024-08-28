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
from torchvision.transforms import v2 as T2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split, Subset
#from torch.utils.tensorboard import SummaryWriter
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

        class_name = None

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

            if class_name is None:
                raise ValueError(f"no objects found in {xml_path}")
            
            boxes_torch = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels_torch = torch.tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes_torch, "labels": labels_torch}
        
            img_path = os.path.join(self.root_path, root.find("filename").text)
            img = Image.open(img_path)#.convert("RGB")
            if self.transforms:
                img = self.transforms(img)

            return img, target


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return tuple(zip(*batch))

train_subset = 0
test_subset = 0

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.compose(transforms)

def loadData():

    #create dataset & split
    dataset = ColumnsDataset("D:\hasti\Object_detection\AUG_images", img_transforms)
    dataset_test = ColumnsDataset("D:\hasti\Object_detection\AUG_images", img_transforms)
    indices = torch.randperm(len(dataset)).tolist()

    dataset_len = len(dataset)
    train_len = int(dataset_len*0.7)
    test_len = int((dataset_len-train_len))

    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    print(len(train_subset))
    print(len(test_subset))

    return train_subset, test_subset

def custom_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train(train_subset, test_subset, img_transforms, model):
    BATCH_SIZE = 8
    #train_dataset = ColumnsDataset()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)

    #testing forward() method to see what the model expects during training
    num_classes=2
    model = custom_model(num_classes)
    dataset = ColumnsDataset("D:\hasti\Object_detection\AUG_images", img_transforms)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0,
        collate_fn = utils.collate_fn
    )

    #for training purposes:
    images, targets = next(iter(data_loader))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.001, weight_decay= 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    model.to(device)

    for images, targets in train_loader:
        print(targets)
        break


    #training loop
    def train_one_epoch(model, optimizer, data_loader, device):
        model.train()
        epoch_loss = 0

        start_time = time.time()
        for i, (images, targets) in enumerate(data_loader):

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

    #train model for n epochs
    num_epochs = 1
    epoch_losses = [] 
    best_loss = float('inf')


    for epoch in range(num_epochs):
        print("----------------- Training Epoch {} -----------------".format(epoch+1))
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        epoch_losses.append(avg_loss)

        lr_scheduler.step(avg_loss)

        print("average loss:", avg_loss)
        print("finished training")
        

    #plot results
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.xticks(range(num_epochs))
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training loss")
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
        torch.save(checkpoint, "best-model-parameters.pt")
    #torch.save(model.state_dict(), 'best-model-parameters.pt')

    return model

def test(model):
    BATCH_SIZE = 8
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, pin_memory=False, shuffle=True, collate_fn=collate_fn)
    #load model with best parameters
    #testing loop

    print("Start testing process")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = custom_model(num_classes)
    checkpoint = torch.load("best-model-parameters.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    image = read_image("D:\hasti\Object_detection\AUG_images\image00404_aug_0.jpeg").convert("RGB")
    eval_transform = img_transforms

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    print(pred)

    #convert images to tensors
    image_tensor = T.ToTensor()(image)
    image_tensor = (224.0 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())).to(torch.uint8)
    image_tensor = image_tensor[:3, ...]
    
    #confidence_treshold = 0.0
    #high_confidence_idx = pred['scores'] > confidence_treshold
    pred_boxes = pred['boxes'].to(torch.int64)
    pred_labels = [f"columns: {score:.3f}" for label, score in zip(pred['labels'], pred['scores'])]
    output_image = draw_bounding_boxes(image_tensor, pred_boxes, pred_labels, colors = "red")
    
    plt.figure(figsize=(12,12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    
    ##create the prediction labels for the images
    #pred_labels = [f"columns: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    #pred_boxes = pred["boxes"].long()  #uitlaten!!!
    #pred_boxes = pred["boxes"].to(torch.int64)
    #output_image = draw_bounding_boxes(image_tensor, pred_boxes, pred_labels, colors = "red")

    #if len(pred_boxes) > 0:
        #output_image = draw_bounding_boxes(image_tensor, pred_boxes, labels=pred_labels, colors = "red")
        #plt.figure(figsize = (12, 12))
        #plt.imshow(output_image.permute(1, 2, 0))
        #plt.axis("off")
        #plt.show()
    #else:
        #print("No bounding boxes found")

    print("Finished testing")
    return model

model = custom_model(num_classes=2)
train_subset, test_subset = loadData()
#train(train_subset, test_subset, img_transforms, model)
test(model)




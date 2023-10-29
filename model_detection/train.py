import torch
import torchvision.transforms as transforms
from dotenv import dotenv_values
from dataset import VOCDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from model import YOLOv1
from utils_copy import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss_copy import YoloLoss
config = dotenv_values(".env")
LEARNING_RATE = float(config["LEARNING_RATE"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE =int( config["BATCH_SIZE"])
WEIGHT_DECAY =int( config["WEIGHT_DECAY"])
EPOCHS = int(config["EPOCHS"])
NUMBER_WORKERS = int(config["NUM__WORKERS"])
PIN_MEMORY = config["PIN_MEMORY"]
LOAD_MODEL = config["LOAD_MODEL"]
LOAD_MODEL_FILE = config["LOAD_MODEL_FILE"]
IMG_DIR = config["IMG_DIR"]
LABEL_DIR = config["LABEL_DIR"]
print(DEVICE)
seed = 123
torch.manual_seed(123)

class Compose(object):
    def __init__(self,transforms) :
        self.transforms = transforms
    def __call__(self,img,boxes) :
        for t in self.transforms:
            img, boxes = t(img),boxes
        return img,boxes
    
transform = Compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train_fn(train_loader, model, optimizer,loss_fn):
    loop = tqdm(train_loader,leave=True)
    mean_loss =[]
    for batch_idx,(x,y) in enumerate(loop):
        # x,y = input, label
        x,y=x.to(DEVICE),y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item()),
        optimizer.zero_grad(),
        loss.backward(),
        optimizer.step(),
        loop.set_postfix(loss=loss.item())
        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = YOLOv1(numberBox=2,numberClass=20,splitFinalSize=7).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay= WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    train_dataset = VOCDataset(
        "data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    test_dataset = VOCDataset(
         "data/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUMBER_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUMBER_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    # if we have pretrain model
    if LOAD_MODEL=='True':
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    for epoch in range(EPOCHS):
        for x, y in train_loader:
           x = x.to(DEVICE)
           for idx in range(8):
               bboxes = cellboxes_to_boxes(model(x))
               bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
               plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

           import sys
           sys.exit()
        # pred_boxes, target_boxes = get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4
        # )

        # mean_avg_prec = mean_average_precision(
        #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        # )
        # print(f"Train mAP: {mean_avg_prec}")
        if epoch ==9 :
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_fn(train_loader,model,optimizer,loss_fn)


if __name__ =='__main__':
    main()

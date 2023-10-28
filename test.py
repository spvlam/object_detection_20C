import torch
from model_detection.utils_copy import intersection_over_union

target = torch.randn((16,7,7,25))
print(target.shape)
predictions = torch.randn((16,7*7*30)).reshape(-1,7,7,30)
print(predictions.shape)
iou1 = intersection_over_union(predictions[..., 21:25],target[...,21:25])
print(iou1.shape)
iou2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
print(iou2.shape)
ious = torch.cat((iou1.unsqueeze(0),iou2.unsqueeze(0)),dim=0)
print(ious.shape)
iouMax, bestBox = torch.max(ious,dim=0)
print("best_box: " ,bestBox.shape)
print("predictions[...,26:30]",predictions[...,26:30].shape)
existedBox = target[...,20].unsqueeze(3)
print("existed_box:", existedBox.shape)
print((bestBox*predictions[...,26:30]).shape)

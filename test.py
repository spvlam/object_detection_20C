import torch
from model_detection.utils_copy import intersection_over_union

target = torch.randint(1,10,(2,2,2,25))

predictions = torch.randint(1,10,(2,2*2*30)).reshape(-1,2,2,30)

iou1 = intersection_over_union(predictions[..., 21:25],target[...,21:25])

iou2 = intersection_over_union(predictions[...,26:30],target[...,21:25])

ious = torch.cat((iou1.unsqueeze(0),iou2.unsqueeze(0)),dim=0)
iouMax, bestBox = torch.max(ious,dim=0)
print("best_box: " ,bestBox)
print("predictions[...,26:30]",predictions[...,26:30])
existedBox = target[...,20].unsqueeze(3)

print(bestBox*predictions[...,26:30])



import torch
def intersection_over_union(first_box,second_box,box_type="corners")->int: 
    """
    parameters:
        first_box( tensor ): (batch_size,4)
        second_box(tensor): (batch_size,4)
        box_type (str) : midpoint/corners if boxes (x,y,w,h) or (x1,x2,y1,y2)
    NOTATION: only support cornners, must convert midpoint to corner
    
    Returns:
        _type_: Intersection over union fraction of first and second boxes
    """
    if box_type == "corners":
        box1_x1 = first_box[0]
        box1_y1 = first_box[1]
        box1_x2 = first_box[2]
        box1_y2 = first_box[3]

        box2_x1 = second_box[0]
        box2_y1 = second_box[1]
        box2_x2 = second_box[2]
        box2_y2 = second_box[3]
    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.min(box2_x2,box1_x2)
    y2 = torch.min(box1_y2,box2_y2)

    intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)
    area_box1 = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    area_box2 = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    print(intersection,area_box1,area_box2)
    return intersection/(area_box1+area_box2-intersection)
def non_max_suppression(boxes,box_threshold,uoi_threshold, box_format="corner"):
    """_summary_

    Args:
        boxes : list[(class, score,x1,y1,x2,y2)] 
        box_threshold : int
        uoi_threshold : int
        box_format (str, optional): _description_. Defaults to "corner".
    return nms array to 
    """
    assert type(boxes) == list
    boxes = [box for box in boxes if box[1] > box_threshold]
    boxes = sorted([box for box in boxes],key=lambda x: x[1],reverse=True)
    boxes_apter_nms =[]
    while boxes:
        choose_box = boxes.pop()
        boxes = [
            box for box in boxes
            if box[0] != choose_box[0]
            or intersection_over_union(
                torch.tensor(box),
                torch.tensor(choose_box),
                box_format
            ) < uoi_threshold
        ]
        boxes_apter_nms.append(choose_box)
    return boxes_apter_nms
def mean_average_pression(boxs,true_box):
    pass
if __name__ == "__main__" :
    # a=torch.tensor([0,0,2,4])
    # b=torch.tensor([1,2,3,6])
    
    # print(intersection_over_union(a,b))
    non_max_suppression([[1,0.9,1,2,3,4],[2,3,1,1,1,1],[3,2,1,1,1,1]],0.5,0.5)
   
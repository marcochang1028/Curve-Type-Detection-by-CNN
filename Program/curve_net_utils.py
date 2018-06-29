from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import imghdr
from PIL import Image, ImageDraw, ImageFont
import colorsys
import random
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import shutil

def iou(box1, box2):
    x,y,h,w = box1
    box1 = np.array([x-w/2, y-h/2, x+w/2, y+h/2])
    x,y,h,w = box2
    box2 = np.array([x-w/2, y-h/2, x+w/2, y+h/2])

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2-xi1,0) * max(yi2-yi1,0)
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_accuracy(Y, Y_pred, num_box_confidence, num_box_coods, num_classes,
                       box_confidence_threshold, iou_threshold):
    box_column_size = num_box_confidence + num_box_coods + num_classes
    Y = np.reshape(Y, [-1,box_column_size])
    Y_pred = np.reshape(Y_pred, [-1,box_column_size])
   
    pos_neg_num = np.sum(sigmoid(Y_pred[np.where(Y[:,0] == 1)][:,0]) < box_confidence_threshold)
    neg_pos_num = np.sum(sigmoid(Y_pred[np.where(Y[:,0] == 0)][:,0]) >= box_confidence_threshold)
    pos_pos_num = np.sum(sigmoid(Y_pred[np.where(Y[:,0] == 1)][:,0]) >= box_confidence_threshold)
    
    temp_Y = Y[np.where(Y[:,0] == 1)]
    temp_Y_pred = Y_pred[np.where(Y[:,0] == 1)]
    temp_Y = temp_Y[np.where(sigmoid(temp_Y_pred[:,0]) >= box_confidence_threshold)]
    temp_Y_pred = temp_Y_pred[np.where(sigmoid(temp_Y_pred[:,0]) >= box_confidence_threshold)]
    
    correct_class_idx = np.equal(np.argmax(temp_Y[:,num_box_confidence + num_box_coods:], 1), 
                                 np.argmax(temp_Y_pred[:,num_box_confidence + num_box_coods:], 1))
    correct_class_num = np.sum(correct_class_idx)
    incorrect_class_num = pos_pos_num - correct_class_num
    
    temp_cood_Y = temp_Y[correct_class_idx][:,num_box_confidence:num_box_confidence + num_box_coods]
    temp_cood_Y_pred = temp_Y_pred[correct_class_idx][:,num_box_confidence:num_box_confidence + num_box_coods]
    
    iou_err_num = 0
    for idx, value in enumerate(temp_cood_Y):
        if (iou(value, temp_cood_Y_pred[idx]) < iou_threshold):
            iou_err_num += 1
    
    accuracy = 1 - ((pos_neg_num + neg_pos_num + incorrect_class_num + iou_err_num) / (len(Y) + 0.))
    return accuracy

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='arial.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c-1]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score[0])

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        bottom, left, top, right = box
        bottom = max(0, np.floor(bottom + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        top = min(image.size[1], np.floor(top + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, bottom), (right, top))

        if bottom - label_size[1] >= 0:
            text_origin = np.array([left, bottom - label_size[1]])
        else:
            text_origin = np.array([left, bottom + 1])

        for i in range(thickness):
            draw.rectangle([left + i, bottom + i, right - i, top - i], outline=colors[c-1])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c-1])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        
def transform_pred_to_boxes_info(y_pred, max_point_range, image_size, box_confidence_threshold):
    
    y_pred = np.reshape(y_pred, [5, 11])
    y_pred = y_pred[np.where(sigmoid(y_pred[:,0]) >= box_confidence_threshold)]
    
    classInfos = y_pred[:,5:]
    classIds = np.argmax(classInfos, 1) + 1
    classValues = np.reshape(np.amax(classInfos, 1), [-1,1])
    scores = sigmoid(y_pred[:, 0:1])
    
    x = y_pred[:, 1:2]
    y = y_pred[:, 2:3]
    h = y_pred[:, 3:4]
    w = y_pred[:, 4:5]
    
    bottoms = image_size * 0.84 - ((((y + h / 2) * (max_point_range**2) * 2.5) + (max_point_range**2) * 2.5) / ((max_point_range**2) * 2.5 * 2)) * (image_size * 0.74)
    lefts = image_size * 0.083 + (x - w / 2) * (image_size * 0.87)
    tops = image_size * 0.87 - ((((y - h / 2) * (max_point_range**2) * 2.5) + (max_point_range**2) * 2.5) / ((max_point_range**2) * 2.5 * 2)) * (image_size * 0.69)
    right = image_size * 0.135 + (x + w / 2) * (image_size * 0.82)
    #bottoms = image_size - 30 - ((((y + h / 2) * (max_point_range**2) * 2.5) + (max_point_range**2) * 2.5) / ((max_point_range**2) * 2.5 * 2)) * (image_size - 50)
    #lefts = 21 - 5 + (x - w / 2) * (image_size - 25)
    #tops = image_size - 25 - ((((y - h / 2) * (max_point_range**2) * 2.5) + (max_point_range**2) * 2.5) / ((max_point_range**2) * 2.5 * 2)) * (image_size - 60)
    #right = 21 + 5 + (x + w / 2) * (image_size - 35)
    
    boxes = np.column_stack((bottoms, lefts, tops, right))
    
    return scores, boxes, classIds

def visualize_predicted_result(logits, img_names, class_names, data_dir, 
                               eval_dir, image_size, max_point_range, box_confidence_threshold):
    class_names = np.array(class_names)
    image_dir = os.path.join(data_dir, eval_dir)
    output_dir = os.path.join(data_dir, "out")
    
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    
    for idx, image_name in enumerate(img_names):
        image_name = image_name[0].decode("utf-8")
        out_scores, out_boxes, out_classes = transform_pred_to_boxes_info(logits[idx], max_point_range, 
                                                                          image_size, box_confidence_threshold)
        print('Found {} boxes for {}'.format(len(out_boxes), image_name)) # Print predictions info
        image, image_data = preprocess_image(os.path.join(image_dir, image_name), model_image_size = (image_size, image_size)) 
        colors = generate_colors(class_names) # Generate colors for drawing bounding boxes.
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors) # Draw bounding boxes on the image file
        image.save(os.path.join(output_dir, image_name), quality=90) # Save the predicted bounding box on the image
        output_image = misc.imread(os.path.join(output_dir, image_name)) # Display the results in the notebook
        imshow(output_image)
    plt.show()
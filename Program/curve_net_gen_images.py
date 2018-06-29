import numpy as np
import matplotlib.pyplot as plt
import os

def gen_initial_params(max_point_range, max_func_num):
    """Generate initial parameters randomly for image generation
    Args:
        max_point_range: the maximum range of the number of points for a curve
        max_func_num: the number of curver types.
    Returns:
        start: randomly selected minimum value for the x (depend on max_point_range)
        end: randomly selected maximum value for the x (depend on max_point_range)
        point_size: the number of the points in a curve
        shiftX: the value that the curve will be shifted in x axis
        shiftY: the value that the curve will be shifted in y axis
        func_index: randomly selected curve type
    """
    start = np.random.uniform(-max_point_range,-5,1)
    end = np.random.uniform(5,max_point_range,1)
    point_size = int(np.round((end - start) * 2))
    shiftX = np.random.uniform(max_point_range, max_point_range * 2, 1) #.make sure x start from 0.
    shiftY = np.random.uniform(-max_point_range**2, max_point_range**2, 1)
    func_index = int(np.round(np.random.uniform(1,max_func_num,1)))
    
    return start, end, point_size, shiftX, shiftY, func_index

def para1(x):
    return 0.2 * x**2 + 0.3 * x

def para2(x):
    return x**2 - x

def para3(x):
    return 0.7 * x**2 + 3 * x


def hyper1(x):
    return np.sqrt(300 + 50 * x**2)

def hyper2(x):
    return np.sqrt(500 + 100 * x**2)

def hyper3(x):
    return np.sqrt(700 + 150 * x**2)

def one_hot_string(max_num, curr_num):
    one_hot_string = ''
    for i in range(max_num):
        if i == (curr_num):
            one_hot_string += '1,'
        else:
            one_hot_string += '0,'
    return one_hot_string[:len(one_hot_string)-1]

def gen_image(img_path, img_num = 10, max_line_num = 5, 
             max_point_range = 10, max_func_num = 6, y_error_range = 1.5, image_size=96):
    """Generate images to simulate the real image.
    
    Args:
        img_path: data directory
        img_num: the number of images will be generated
        max_line_num: max number of curves in an image
        max_point_range: max range of number of points will be generated in a curves 
                        (basic number of points is 10 and total number of data points 
                        of a curve is "basic number + max_point_range*2")
        max_func_num: the number of curver types.
        y_error_range: add noisy to the y values of curvers
        image_size: image height and width
    """
    dataset = []

    image_inch = image_size / 96.
    marker_size = image_size / 200.
    
    # Start to generate images
    for img_idx in range(img_num):
        
        #. Randomly select the number of curves which will be generated in an images (each image will have different number of curves)
        line_num = int(np.round(np.random.uniform(1,max_line_num,1)))
        
        fig = plt.figure(figsize=(image_inch,image_inch), dpi=96)
        # As we want to sort the curves by its start point value for each image, we need to store the curves in a temp array
        one_img_text_arr = np.zeros([5,11])
        # Store the info in a string and write it into the csv later.
        one_img_text = str(img_idx) + '.png,' 
        
        # Start to generate curves in an image
        for line_idx in range(max_line_num):
            # If the number of generated curves smaller than the max curves number in this image than continue to generate next curve.
            if line_idx < line_num:
                one_img_text_arr[line_idx,0] = 1
                
                #. Generate some random parameters for the curve generation
                start, end, point_size, shiftX, shiftY, func_index = gen_initial_params(max_point_range, max_func_num)
                
                #. Generate the values of x
                x = np.random.uniform(start, end, point_size)
                #. Generate the values of y based on x and randomly selected curve type
                if func_index == 1:
                    y = para1(x)
                elif func_index == 2:
                    y = para2(x)
                elif func_index == 3:
                    y = para3(x)
                elif func_index == 4:
                    y = hyper1(x)
                elif func_index == 5:
                    y = hyper2(x)
                elif func_index == 6:
                    y = hyper3(x)
                
                # Add noise to the values of y
                error = np.random.uniform(-y_error_range, y_error_range, point_size)
                y = y - error #let the points not exactly on the trace
                # Shift the poisiton of curve
                x = x + shiftX
                y = y + shiftY
                
                plt.plot(x, y, 'ro', markersize=marker_size)
                
                # Normalize it by the scale of x and y axes
                x = x / (max_point_range * 4.)
                y = y / ((max_point_range**2) * 2.5)

                # Calculate the box information for a curve (middle point of (x,y) and box width and hight)
                x_max = np.max(x)
                x_min = np.min(x)
                y_max = np.max(y)
                y_min = np.min(y)
                box_hight = y_max - y_min
                box_width = x_max - x_min
                box_mid_x = box_width / 2 + x_min
                box_mid_y = box_hight / 2 + y_min
                
                one_img_text_arr[line_idx,1] = box_mid_x
                one_img_text_arr[line_idx,2] = box_mid_y
                one_img_text_arr[line_idx,3] = box_hight
                one_img_text_arr[line_idx,4] = box_width
                one_img_text_arr[line_idx,5+func_index-1] = 1 #.one hot vector for the cuver type
        
        #Sort the curve by column1 then column to create some causal relationship 
        #between box sequence and coodination. After sorting proceduce, pad it with 0 to fit 5 boxes.
        tempCurves = one_img_text_arr[np.where(one_img_text_arr[:,0] == 1)]
        tempCurves = tempCurves[tempCurves[:,2].argsort()]
        tempCurves = tempCurves[tempCurves[:,1].argsort()]
        for c in tempCurves:
            for t in range(11):
                one_img_text += str(c[t]) + ','
        nonCurves = one_img_text_arr[np.where(one_img_text_arr[:,0] == 0)]
        for c in nonCurves:
            for t in range(11):
                one_img_text += str(c[t]) + ','
        
        dataset.append([one_img_text[:len(one_img_text)-1]]) #remove last comma and add into dataset
        
        # Save the image
        plt.axis([0, max_point_range * 4, -(max_point_range**2) * 2.5, (max_point_range**2) * 2.5])
        plt.savefig(os.path.join(img_path, str(img_idx) + '.png'))
        plt.close(fig)
    
    # Write all image information into a csv file
    with open(os.path.join(img_path,'data.csv'), "w") as output:
        for text in dataset:
            output.write("%s\n" % text[0])
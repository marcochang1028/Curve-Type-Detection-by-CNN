# Curve Type-Detection-by-CNN
An ongoing research project which uses CNN to detect the type and the position of curves in images.

## Objective
In this project, the CNN model needs to detect the type and the position of curves which are made up of a group of points (not a line). Since I am unable to get enough data for the training, I wrote a program to generate the simulated images. Next, I follow [`CIFAR-10`](https://www.tensorflow.org/tutorials/deep_cnn) to construct the data pipeline and the structure of the functions and files. I also borrow the idea from YOLO V2 model to design the structure of labels (box information).  

Please note that it is an ongoing project, thus, the model is still being optimized, especially the layers, parameters, loss function. 

## Requirements
- Python 3.6
- Tensorflow 1.8

## Data
The image generation program is in the file `curve_net_gen_images.py` in the folder `Program`. The functions include:
- Randomly generate 1 to 5 curves in the image.
- Randomly generate 6 types of curves (3 types of the parabola and 3 types of the hyperbola).
- Randomly shift the position of curves in the image.
- Randomly generate a different number of data points for a curve and add some noise to the coordinate of data points.
- Generate a CSV file to store the curve information for each image which can be used to train the model.

I put a zip file in the folder `Data` with 512 images just for reference (if you really want to train this model, it needs at least 100000 images). Also, when you run the model, the program will generate the data for you, if you do not have any simulated images in the `temp` folder.

## Program
- `curve_net_gen_images.py` - Generate simulated images.
- `curve_net_input.py` - Reads simulated images (implements data pipeline).
- `curve_net.py` - Builds the CurveNet model (implements the layers and the loss function).
- `curve_net_train.py` - Trains a CurveNet model.
- `curve_net_eval.py` - Evaluates the predictive performance of a CurveNet model.
- `curve_net_utils.py` - The helper functions to calculate accuracy (IOU) and visualize the predicted results (draw the boxes on image)

## References
- [Convolutional Neural Networks (CIFAR-10)](https://www.tensorflow.org/tutorials/deep_cnn)
- [YOLO V2 model](https://pjreddie.com/darknet/yolo/)
- [Convolutional Neural Networks (online course)](https://www.coursera.org/learn/convolutional-neural-networks)

## Remarks
Welcome to contact me via [marcochang1028@gmail.com](mailto:marcochang1028@gmail.com)

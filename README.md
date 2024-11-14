This GitHub repository page contains the code files that were used during my graduation project. 

There are three files in total.
I. An image augmentation file that can be used to expand your dataset. This file makes various changes to your images such as rotating them and changing the colors.
II. A code for object detection, this code was during the thesis to train a Faster R-CNN model on power socket images, using a transfer learning method. In addition to the training, the model adds an overlay to the sockets which are used for comparison reasons between the predictions and the placement in the BIM model. This code also includes AP calculations at IoU50, and adds an as-built property to an IFC model using IfcOpenShell. Instructions are provided within this file.
III. The Open3D code can be used to create point cloud images from RGBD camera data. You will need an RGB image and the corresponding depth map.

The labeled socket dataset that was used for the object detection code will be added to this repository soon.

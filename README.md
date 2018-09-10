
# Implementation of object detection for custom images and videos using Tensorflow API  and Azure services
This repository presents two approaches for implementation of custom object detection for images and video (video comes later) using deep learning.
1. Using Azure Machine Learning Package for Computer Vision (AMLPCV) 
https://docs.microsoft.com/en-us/python/api/overview/azure/computer-vision?view=azure-cv-py
2. Using Tensorflow Object Detection library (TF)
https://github.com/tensorflow/models/tree/master/research/object_detection
AMPCV is the library developed by Microsoft that tightly integrates with the old Machine Learning Workbench for deployment. This library itself bases on implementation of the Tensorflow Object Detection Libary for Faster R-CNN
### Why tensorflow approach?
- While AMLPCV is an excellent high level library to quickly develop and deploy highly accurate object detection model, it has several limitations: 
    - Use old ML workbench as development and deployment platform which will be deprecated soon
    - Only support Resnes50 as the pretrained model. Different scenarios work better with different pretrained model 
    - Does not provide a lot of flexibility for configurations like TF framework
    - Reliance on ML workbench makes it very difficult to run this with other Python packages such as Spark API due to versions conflicts (required TF 1.4...) so batch scoring in production is very difficult
    - AML workbench justs support ACI which is not a great option for production deployment
- Using TF for object detection while more complex due to many steps but it gives you lots of customizability for model training and deployment 

Overview of steps:
1. Provision of GPU DSVM 
  - For AMLPCV you should choose windows because it relies on AML workbench
  - For Tensorflow implementation ubuntu is recommended
2. Annotation of objects in images
  I like label image https://github.com/tzutalin/labelImg because of its simpicity.
  In custom object detection you do not normally have a lot of sample images so my best practice is try to draw multiple bboxes for each object, each is a little bit different to generate more data for training.
  My implementation bases on the assumption that the output xml files and images files are in the same directory and have same name for each image (of course different extension)
3. Installation of libraries
 Please follow the instruction to install AMLPCV, Tensorflow object detection in the DSVM. If you want to deploy the trained model (only works wih tensorflow) using new AML API to AKS cluster then please register with ViennaDocs team to get their preview version of API.
4. Training and model selection
Run training with different params and choose the best performing model
5. Deployment
Deploy to ACI or AKS. Batch deployment for batch scoring coming later
Copyrights: Caution do not share data and trained model outside Microsoft. The model is trained based on customer provided data
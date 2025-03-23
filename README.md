# Brain-Tumor-Deep-Learning-Detection-CNN-VGG-ResNet50-EfficientNet-ConvNeXt-and-Segmentation-SAM-UNet

![Harvard_University_logo svg](https://github.com/user-attachments/assets/0ea18127-d8c2-46ec-9f3e-10f2dc01d4d7)

![Harvard-Extension-School](https://github.com/user-attachments/assets/7de8c00d-6d74-456f-9b18-abb3174e83d5)

## First Words

In my **Brain Tumor Detection** project, I chose this dataset from Kaggle with 115 brain imges with tumor and 98 without (clearly showing imbalance): 

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Timeline: January 6th - May 16th, 2025

## Professor: Stephen Elston

## Author: Dai-Phuong Ngo (Liam)

## Course: CSCI E-25 Computer Vision in Python

First, I will start with **Data Exploration** and **Image Preprocessing**:

For **Data Exploration**, I will try out the CDF and histogram distributions on 3 standard colors or predefined colors (3 to 5) to examine the color distribution by objects in typical MRI images: black background, skull layer, brain tissues, fluids, brain cord, tumor (if any) based on different brightness, contrast, sharpness, quality, angle of MRI scanning slice, unnecessary objects (eyes, neck, etc).

For **Image Preprocessing**, this phase serves 2 purposes of my 2 end goals:

1/ Prepare images for binary classification:

I will apply different techniques for image cropping (background might be too wide and unnecessary), skull layer removal (fully or partially or none removed by **Morphological Snakes** and/or **Morphological Filtering** for largest contour detection and removal), depending on image quality and close-to-global techniques), contrast enhancement, sharpness adjustment, noise surpression in predefined order. 

https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_morphsnakes.html

https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html

Then I will try out further Watershed Segmentation to highlight different regions in the brain images (with/without tumor) in colors for data enrichment before moving to data augmentation and class rebalancing techniques to diversify and balance data.
https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_watershed.html

2/ Prepare images with segmented tumor:

For those images with tumor, I will apply this first half part of this technique **Segment Anything Model (SAM) with Transformers** to generate color-highlighted tumor regions and build a separate annotated dataset for fine-tuning segmentation models.
https://keras.io/examples/vision/sam/

My project has 2 predictive end goals: 

1/ **Binary Classification** - Predict whether a brain MRI image shows a tumor:

I will start with a custom **CNN** and then apply **Transfer Learning** with models such as **VGG16, ResNet50, EfficientNetV2, and ConvNeXt** (more preferrable) variants to compare performance across metrics beyond accuracy.
https://keras.io/api/applications/

2/ **Tumor Segmentation** - Highlight the tumor region in MRI images where present: 

With the #2 annotated data thanks to SAM as mentioned above, I will fine-tune the pretrained model, such as **SAM** and/or **Image segmentation with a U-Net-like architecture**.

https://keras.io/examples/vision/oxford_pets_image_segmentation/

This dataset is manageable in size, and I’ll rely on Keras, scikit-image and pretrained models to optimize both performance and training time.

# 🧠🧬 Brain Tumor Deep Learning Detection (CNN, VGG, ResNet, EfficientNet, ConvNeXt) and Segmentation (Segment Anything Model (SAM), UNet-like Architecture, DeepLabV3+, K-Means)

![Harvard_University_logo svg](https://github.com/user-attachments/assets/0ea18127-d8c2-46ec-9f3e-10f2dc01d4d7)

![Harvard-Extension-School](https://github.com/user-attachments/assets/7de8c00d-6d74-456f-9b18-abb3174e83d5)

## **Master of Liberal Arts, Data Science**

## CSCI E-25 Computer Vision in Python

## Timeline: January 6th - May 16th, 2025 (In Progress)

## Professor: Stephen Elston

## Author: Dai-Phuong Ngo (Liam)

## Executive Summary

When this academic term of Computer Vision was about to arrive, I had been considering which kind of project and data I should invest in and take privileges from this course and Professor Stephen Elston's professional advice. I used to work with non-structured data, such as images or PDF files, to extract necessary information for later process and insights. The good thing about them is that they tend to be black and out with text and figures for extraction and identification. For those projects in my previous company, I utilized the power of Computer Vision programming and efficient models in Python for absolute solutions without relying on paid AI services from big tech firms. Later on, I realized that I should try to work on more complex image data that is not about text and figures on digital files and physical papers for this course's project. There are multiple reasons for me in doing so: pushing myself limits out of the comfort zone of the text-typed image data into very unstructured data, immerse myself into a new industry and reasearch, apply new kinds of algorithms, metrics and Deep Learning models for computer Vision purposes.

Tumors in brain is the most dangerous kinds of tumors worldwide. Glioma and Meningioma are the two most popular cases of brain tumors, especially Glioma. Based on reasearch, the mean survial time after diagnosis is fewer than 14 months for studied patients. MRI, which is Magnetic Resonance Imaging, is commonly leveragted by medical experts in determining brain tumor potential as it provides a wide varierty of contrasts from brain tissues in every imaging modality. Therefore, we can assume that one single patient conducting an MRI scanning might have multiple MRIs for their own brain, which will be then manually diagnosed by specialists. This kind of manual diagnosis is significantly time-consuming for every patient and only be executed by neroradiologists. Consequently, with modern and developed deep learning models, this diagnosis process can be automated with classification and segmentation to provide insights timely for further diagnosis and treatment, minimizing risks of growing neurological orders, e.g. ALzheimer's disease (AD), dementia, and schizophrenia. My project aims to propose an end-to-end deep learning pipeline for automated brain tumor detection and segmentation using the assembled MRI images. My approach concentrates on ehancing grayscale brain MRIs by a sophisticated technique studied from the CSCI E-83 course by Professor Oleg S. Piyanykh, PhD, Medical Analytics Group, Department of Radiology, Massachussetts General Hospital, Harvard Medical School, named as MUSICA (Multiscale Image Contrast Amplification) to highlight different regions in MRIs and improve downstream model performance, comprised of Classification and Segmentation, using state-of-the-art deep learning models.

## First Words

In my Brain Tumor Detection project, I chose this dataset from Kaggle with 115 brain imges with tumor and 98 without (clearly showing imbalance) and some of those without tumor do actually have tumor, which createa even more imbalance that I will explain and resolve later:

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

First, I will start with Data Exploration and Image Preprocessing:

For Data Exploration, I will try out the CDF and histogram distributions on 3 standard colors or predefined colors (3 to 20) to examine the color distribution by objects in typical MRI images: black background, skull layer, brain tissues, fluids, brain cord, tumor (if any) based on different brightness, contrast, sharpness, quality, angle of MRI scanning slice, unnecessary objects (eyes, neck, etc). Most of the images in my current dataset comprise of Axial brain MRIs containing Glinoma, Pituitary, Menigioma. Only some images are Coronal images.

![cancers-15-04172-g001](https://github.com/user-attachments/assets/f7e97196-57fa-4aae-b58d-eabe1a04ca0e)

*Abdusalomov, A. B., Mukhiddinov, M., & Whangbo, T. K. (2023). Brain Tumor Detection Based on Deep Learning Approaches and Magnetic Resonance Imaging. Cancers, 15(16), 4172. https://doi.org/10.3390/cancers15164172*

For Image Preprocessing, this phase serves 2 purposes of my 2 end goals:

1/ Prepare images for binary classification:

I will apply different techniques for image cropping (background might be too wide and unnecessary), skull layer removal (fully or partially or none removed by Morphological Snakes and/or Morphological Filtering for largest contour detection and removal), depending on image quality and close-to-global techniques), contrast enhancement, sharpness adjustment, noise surpression in predefined order.

https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_morphsnakes.html

https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html

Then I will try out further Watershed Segmentation to highlight different regions in the brain images (with/without tumor) in colors for data enrichment before moving to data augmentation and class rebalancing techniques to diversify and balance data. https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_watershed.html

2/ Prepare images with segmented tumor:

For those images with tumor, I will apply this first half part of this technique Segment Anything Model (SAM) with Transformers to generate color-highlighted tumor regions and build a separate annotated dataset for fine-tuning segmentation models. https://keras.io/examples/vision/sam/

My project has 2 predictive end goals:

1/ Binary Classification - Predict whether a brain MRI image shows a tumor:

I will start with a custom CNN and then apply Transfer Learning with models such as VGG16, ResNet50, EfficientNetV2, and ConvNeXt (more preferrable) variants to compare performance across metrics beyond accuracy. https://keras.io/api/applications/

2/ Tumor Segmentation - Highlight the tumor region in MRI images where present:

With the #2 annotated data thanks to SAM as mentioned above, I will fine-tune the pretrained model, such as SAM and/or Image segmentation with a U-Net-like architecture.

https://keras.io/examples/vision/oxford_pets_image_segmentation/

https://keras.io/examples/vision/deeplabv3_plus/

https://keras.io/examples/vision/basnet_segmentation/

https://keras.io/examples/vision/fully_convolutional_network/

This dataset is manageable in size, and I’ll rely on Keras, scikit-image and pretrained models to optimize both performance and training time.

## Problem Statement

Brain tumor detection from MRI scan images is my project for this course to explore and apply feature extraction techniques to identify tumors more efficiently and accurately. These images include different sizes and shapes of tumors and normal brain scans for a balanced dataset. Some features for tumor detection that I have been thinking about, such as:

- Tumors have different textures than normal brain tissue and asymmetry location. Extracting features based on entropy can differentiate these areas.

- Tumors have unusual shapes, and boundaries. Using edge detection techniques like Sobel filters and Canny edge detection can help to capture them.

- Tumors can be displayed in bright or dark areas. Using intensity histograms and threshold techniques can highlight abnormal areas.

Some key properties that I will use in these applications are as follows:

- Asymmetry and unusual boundaries: tumor objects tend to exist on one side, causing an asymmetry appearance.

- Contrast differences: tumors can be displayed with contrast variations which is not observed in health brain image.

- Different scale analysis: Tumors can exhibit in varying sizes so features should be extracted at different scales when analyzing multi-scale features.

Another major concern is about Bias. Bias in the medical imaging powered by machine learning, particularly brain tumor detection is a major concern. One common issue is because of the quality of the images, which are used for training purposes, that can affect the precision and reliability of classification, segmentation and even object detection models.

With that being mentioned, I find the key problem that is the most challenging for image processing phase is image quality variation. The scanner's images can experience low resolution (the most obvious behavior), noise, scanner artifacts (they can show a character like A, B, C at the image corner) and contrast inconsistencies for the whole brain and skull artifacts. As the images vary in different aspects, which I consider them as exceptional cases from standard and common quality brain images: brightness, resolution and angles (from top to bottom, or from back to front, with or without eyes and neck, which adds up more artifacts than usual), that accumulates with more challenges and struggles when dealing with feature extraction, causing inconsistent predictions.

Another major issue is class imbalance in the given images. In my dataset, there are 98 images of brains without tumors versus 155 images of brains with tumors. This imbalance causes the models to be trained with and observe more examples of tumors, which sounds attractive, but unfortunately, might be more prone to and biased toward detecting tumors, even when there is no tumor exists, which is false positive. On the other hand, underrepresented images without tumors might cause tumors undetected, which is false negative, causing more serious risk.

To rectify the problems, I've been thinking about some techniques to apply for and test the ending results when put into the models:

- Data augmentation can help to apply certain tasks like rotation, flipping, contrast adjustment and noise surpression to have more diverse data for model training.

- Class rebalancing might offer an effective fight against bias with oversampling or undersampling. I have to think which one makes more sense for this project.

- Transfer learning can reduce much of model development by using pretrained models that were trained on large-scale medical datasets to improve performance on my limited data.

- Not relying on a single metric, accuracy, is a must as there are other metrics like precision, recall and their curves, AUC ROC, F1 to consider about.

## Modelling, Algorithm and Evaluation Strategy

The downstream pipeline serves two purposes: classifcation and segmentation. Regarding the Classification, the models will predict whether a brain MRI contains a tumor or multiple tumors. And regarding the Segmentation, the models will locate tumor regions with a segmented mask. Both tasks will be built on top of data-augmented MUSICA-enhanced grayscale images which are preprocessed for improving contrast and fine details. For segmentation, the models will definitely need masked tumor-relevant images that I will deploy the available SAM to prepare segmented images on top of the MUSICA-enhanced and cropped MRIs. I will research and implement different versions of each model, if available, for the deep learning architectures, including the models of ResNet, EfficientNet, ConvNet for Classification and SAM v2, U-Net for Segmentation using Keras, Tensorflow and Pytorch frameworks. I assume that with the combined efforts of image processing, quality enhancement and powerful, up-to-date deep learning models, the whole architectures can strive to improve accuracy, interpretability as well as robustness when facing with limited labeled data in medical imaging. I target my final product to be an automated solution to diagnose tumor with binary tumor detection and precise tumor masks for further expertise's analysis and decision.

1. Classification Models in Keras:

- ResNet is a good first option for image classication as the datasets I have from Kaggle are labeled in the image file names, but not labeled or annotated in any way on or within the image. This model tends to offer residual skip connection to handle vanishing graident. Therefore, I choose this model and some of its versions as my baseline.

- Efficient is the next efficient option for this purpose with optimal scalin in depth, width and resolution. It can serve as the second backbone for my archtecture.

- ConvNeXt is the most modern, state-of-the-art deep learning model for quick experimentation and visualization.

These classification models are expected to perform binary classification on testing image data without annonating any label or class after prediction directly on the images.

2. Segmentation Models in Keras/Pytorch, Tensorflow, HuggingFace:

- U-Net serves as the backbone and standard model for my second purpose, segmentation which is mainly used for medical imaging.

- SAM v2 (Segment Anything Model) is a powerful zero-shot segmentation model that can serve 2 purposes: guided segmentation for providing newly labeled data, which I don't have from the original dataset, with precise masks for later training, and unguided segmentation with fine-tuning capabilities, if I can decide later to dive deeper in this path of tuning it.

- Fully Convolutional Network (FCN) is an early CNN-based segmentation model with fully convolutional layers. This is an additional model to segment image without flattening.

- DeepLabV3+ is a state-of-the-art segmentation architecture beside these two to segment tumor boundaries and handle fine details as well as multi-scale information.

- BASNet is a boundary-aware segmentation which offers dual-stream architecture. I find it appropriate for further research and implementation for tumor boundary refinement and edge-focused segmentation.

All of these segmentation models are expected to capture complex tumor edges accurately while improving localization of tumor borders and handle small or blurred tumor regions better.

## Data Flow Pipeline

**Phase 1:**

Original MRI Images → Crop Brain Region & Preprocess MRIs → MUSICA Enhancement → Masked Images as Labeled Data → Data Augmentation

**Phase 2:**

→ Classification Model (Tumor / No Tumor)

→ Segmentation Model → Predict Tumor Mask

## Evaluation Strategy

All models after research, testing and application are compared via these metrics:

1. Classification:

- Accuracy: it calculates general proportion of correct predictions of either tumor or not. But in case of imbalanced data, this metric can not be fully trustred as misleading sense might be given.

- Precision: this assesses how many of the predicted tumor images are truly tumors, which might lead to unnecessary alarm if given false positives.

- Recall: this assesses how many actual tumor images are correctly identified. Therefore, it might lead to missing a tumor with a false negative, which is worse than the unnecessary alarm of false positives.

- F1-score: this balances both above concerns: over prediction and missing tumors.

- ROC-AUC Curve: this evaluates the model capabilities in differentiating between tumor and no tumor across thresholds.

- Confusion Matrix: this helps with breakdown in number of true and false positives and negatives by error types.

- GradCAM Visualization: this gives a heatmap to interpret which image part was concentrated during training and prediction, which is beneficial to verify the targeted regions to be put into model's consideration.

2. Segmentation:

- Dice Coefficient: this serves as the main metric for tumor region overlap.

- IoU (Intersection over Union): it checks the pixel agreement.

- Pixel Accuracy: it assesses the proportion of correctly classficed pixels out of all pixels in an image.

- Boundary F1 Score: it assesses sharpness of mask boundary.

- Visual Inspection of mask quality: it check how clean the boundaries of tumor are.

- Evaluate on multi-region tumors: this final step compares the performance of all models on how well each model can mask the tumor regions if one brain contains two or more tumors, instead of only one tumor in most cases.

In my project, optimizing hyperparameters is what I will conduct to maximize model performance for my models, especially the classification. For segmentation, that would require more time researching the provided hyperparameters to upgrade or fine-tune them. Therefore, I will apply different strategies depending on the model type and complexity.

1. Classification with Resnet, EfficientNet, ConvXNet:

The hyperparmeters and proposed search space I plan to initially implement once selecting the certain version of each model are: Learning Rate (1e-5, 1e-4, 1e-3, 1e-2), Optimizer with common Adam, RMSprop, SGD, DropRate with 0.2 to 0.5 by 0.1 increment, Dense Layer Size by 128, 256, 512 (which is the largest size as increasing it might complicate the model processing), batch size with 16, 32 and 64, as well as data augmentation techniques flipping (as tumor can exist on either left or right brain parts), rotating (by a decent degree as patient must lay down while mainintaing straight up body when taking MRI shots) and brightness (as MRI scanners export varied image quality and hence brightness). Some search methods are also considered to put into practice with Random Search for quick exploration or Grid Search for better tuning.

2. Segmentation with U-Net, SAM v2, FCN, DeepLabv3+, BASNet:

Learning Rate, Optimizer, Batch Size, Augmentation and Number of Filters are quite the same as the Classification models. In addition, the segmentation will have Loss Function of Dice Loss, Binary Crossentropu and Focal Loss.

In my whole project, I aim for time allocation of these tasks as follow by leveraging Google Colab Pro+ with GPU accelerated training for Keras models:

- Data Exploration, Preprocessing by 10%

- MUSICA Enhancement, Masking for labeled data, Augmentation by 10%

- Classification and Tuning by 30%

- Segmentation and Hyperparameter Tuning by 35%

- Analysis, Evaluation and Visualization by 10%

- Report Writing and Finalization by 5%
  
## Computer Vision Application for Image Preprocessing

For my project, I plan to work on brain tumor detection in grayscale MRI images using deep learning techniques such as segmentation, augmentation, edge sharpening and if possible, masking and/or segmentation. MRI images are grayscale and vary as they depend on scanners, capturing different slices of the brain. These challenges arise when working with MRI data that could lower model performance. An issue is a lack of high-quality MRI images, especially annotated ones, causing an imbalance in the dataset. In addition, there are variations in brain size, tumor size and the angle of MRI scan which also lower model accuracy. To improve image quality and, hopefully, consistency, some preprocessing techniques might help as I try to increase model accuracy. The first ones can be normalization and standardization as pixel intensity values can be adjusted to a common scale. Enhancing contrast by histogram equalization can highlight tumor areas more effectively. Then, applying filters like Gaussian can denoise and reduce artifacts. As the image data is not abundant for training, data augmentation like rotation, flipping or even shifts in intensity can diversify the data and enhance the generalization of models. Edge detection and sharpening can also improve tumor boundaries. Before that, removing unnecessary areas of the images can help crop the image by covering the skull only.

## Major Concern - Bias

Bias in the medical imaging powered by machine learning, particularly brain tumor detection is a major concern. One common issue is because of the quality of the images, which are used for training purposes, that can affect the precision and reliability of classification, segmentation and even object detection models.

With that being mentioned, I find the key problem that is the most challenging for image processing phase is image quality variation. The scanner's images can experience low resolution (the most obvious behavior), noise, scanner artifacts (they can show a character like A, B, C at the image corner) and contrast inconsistencies for the whole brain and skull artifacts. As the images vary in different aspects, which I consider them as exceptional cases from standard and common quality brain images: brightness, resolution and angles (from top to bottom, or from back to front, with or without eyes and neck, which adds up more artifacts than usual), that accumulates with more challenges and struggles when dealing with feature extraction, causing inconsistent predictions. 

Another major issue is class imbalance in the given images. In my dataset, there are 98 images of brains without tumors versus 155 images of brains with tumors. This imbalance causes the models to be trained with and observe more examples of tumors, which sounds attractive, but unfortunately, might be more prone to and biased toward detecting tumors, even when there is no tumor exists, which is false positive. On the other hand, underrepresented images without tumors might cause tumors undetected, which is false negative, causing more serious risk.

To rectify the problems, I've been thinking about some techniques to apply for and test the ending results when put into the models.

- Data augmentation can help to apply certain tasks like rotation, flipping, contrast adjustment and noise surpression to have more diverse data for model training.

- Class rebalancing might offer an effective fight against bias with oversampling or undersampling. I have to think which one makes more sense for this project.

- Transfer learning can reduce much of model development by using pretrained models that were trained on large-scale medical datasets to improve performance on my limited data.

- Not relying on a single metric, accuracy, is a must as there are other metrics like precision, recall and their curves, AUC ROC, F1 to consider about.

I've been thinking about finding more images to merge into my dataset or find a brand new dataset but their load are very heavy for limited cloud storage and model consumption within limited project timeline for this course if not running on Kaggle notebook but on Google Colab Pro (plus) and better version of Google Drive.

## Major Preparation - Feature Extraction

Brain tumor detection from MRI scan images is my project for this course to explore and apply feature extraction techniques to identify tumors more efficiently and accurately. These images include different sizes and shapes of tumors and normal brain scans for a balanced dataset. Some features for tumor detection that I have been thinking about, such as:

Tumors have different textures than normal brain tissue and asymmetry location. Extracting features based on entropy can differentiate these areas. 

Tumors have unusual shapes, and boundaries. Using edge detection techniques like Sobel filters and Canny edge detection can help to capture them.

Tumors can be displayed in bright or dark areas. Using intensity histograms and threshold techniques can highlight abnormal areas.

Some key properties that I will use in these applications are as follows:

- **Asymmetry and unusual boundaries**: tumor objects tend to exist on one side, causing an asymmetry appearance.

- **Contrast differences**: tumors can be displayed with contrast variations which is not observed in health brain image.

- **Different scale analysis**: Tumors can exhibit in varying sizes so features should be extracted at different scales when analyzing multi-scale features. 

## Cropping Images while reducing cropping aggressiveness

I will approach the brain MRI images without tumor first to validate if all images truly contain no tumor as wrongly placed or classified images could diminish the training process later on. My purposes at this step are to crop the brain region from each image to remove irrelevant background, including: excessive black background, non-black surrounding frame, some colored annotations on the black background which tend to exist at the image corner. At this stage, I decided not to engage with skull layer removal as each image might have inconsistent qualities, brightness, sharpness, resolution, etc which hinders me to find an absolutely efficient skull layer removal for all images. Then I will compute RGB Cumulative Distribution Functions (CDFs) from the cropped images and visualize both the cropped brain regions and their RGG intensity profiles. These plots can create a visual and statistical baseline for healthy brain understanding and comparison with brain images suffering with tumor. 

Then I will switch my attention to the MRIs with tumor. Similarly to the techniques applied for those MRIs without tumor, I'll make the cropping more conservative by blurring less before thresholding to preserve edges and using adaptive thresholding instead of Otsu for more local sensitivity, as well as expanding the bounding box slightly after detecting the largest contour. This makes sure that I will not clip out vital anatomy, especially the skull later and some of brain tissue which might have tumor closely attached to. I identified some images in this subset that are the minority needing soft cropping only.

### Crop Function

As images come from different MRI scanners and scanning angles with varied qualities, resolution, sizes, etc, I will isolate the actual brain content in the image and remove everything else like background, text overlays, non-black edges and excessive noise. My aim is to leave the new edges of the cropped images that can fit the skull outer layer. Therefore, later on, when applying any other processing steps, the cropped images will be less affected by those artifacts.

I came up with two cropping strategies:

- Soft cropping uses adaptive thresholding: I believe this will be useful for complex or inconsistent lighting conditions in minority of MRIs. This technique calculates a threshold value for each region in the image separately. Therefore, it upholds better segmentation when lighting is uneven or image quality is unreliable.

- Hard or standard cropping uses Otsu’s global thresholding: This technique is ideal for well-contrasted, evenly lit images, comprised of by the majority of MRIs.

When either way of the thresholding step is applied, I can extract contours, which are conencted white areas, from the binary image. At this stage, I assume the largest contour represents the brain object. Then I wil compute a bounding box around it. I also add padding to minimizing risks of cutting off any part of the brain, especially the brain tissues inside the skull layer. Another way is that soft cropping is also helpful to manually pinpoint any images that are cut off too harshly by the hard or standard cropping. Eventually, this step makes sure that the brain is centered and I can remove other irrelevant or non-crucial areas before further analysis.

### RGB CDF

At this step, I will compute the cumulative distribution of pixel intensity values for each color channel of blue, green and red. For each channel, I will flatten the 2D image into a 1D liste of pixel values and then calculate a histogram of intensity counts with 256 bins for intensity levels ranging from 0 to 255 on the x axis. After that, I calcualte the cumulative sym of the histogram, which provides the CDF. At last, I normalize the CDF by dividing it by the total number of pixels to contain the range from 0 to 1 on the y axis. I believe the CDF is the best option to illustrate the pixel proportion to understand the MRIs' contrast, brightness and color balance.

### Batch Processing and Visualization

For easiness and readability of viewing the images and CDFs, I chose batches of 10 at a time to be processed. For each batch, there will be 10 images processed. And for each image, I will check if soft cropping is better selected than hard cropping before applying cropping to isolate the brain. Then I compute the RGD CDFs as explained above for the cropped region. Eventually, these processes are crucial for medical imaging as if a tumor is diagnosed, there migth be increased variance, skewed brightness or even clusters of very dark or very bright regions. CDF also provides me detection whether the image enhancement (MUSICA to be applied later) or cropping approaches might change characteristics in the image.

### Touch up on Image 44

Based on the above results, I noticed that the Image 44 is an outlier with the white background while all other images have black background. I will perform a background cleanup operation to detect and replace white background pixels with black ones. Below is the visual comparision of the original and the result. Then I will overwrite the original with the result in the subfolder. This step is to standardize the image background as an outlier like this might cause distractions, and to prevent white regions from making pixel intensity distributions skewed or compromising my later segmentation algorithms. The white background might be problematic for image segmentations as white pixels might be wrongly interpreted as oart of the tissue or especially of the tumor. It will also affect the CDFs when white pixels inflate the intensity profile. Furthermore, there might be risk that white background can cause sensitivity for CNNs trained on natural MRIs (having black background).

![download (73)](https://github.com/user-attachments/assets/1d29ab6e-3f8d-43d3-be70-ca92504b4c3d)

### Resolve white layer surrounding Image 145's black background

Likewise, I also notice another outlier that have white border which requires turning these white border pixels into black. The reason is that they can inferfere with image analysis like enhancement, clustering and segmentation. 

My idea starts with converting it to grayscale and create a white pixel mask, which is not directly used for masking but for diagnostic step. I then create a mask using cv2.inRange() to select pixels falling into the RGB ranges between [220, 220, 220] for light gray and near white pixels, and [255, 255, 255] for pure white pixel. The mask serves as an identification of potential white border artifacts or background which need to be blacked out. After that, I use a small kernel of 3x3 square to dilate the mask which expands the white area slightly. This add-on helps to catch thin outlines or isolated white pixels. After identifying the mask of unnecessary white areas, I will replace them with black [0, 0, 0]. This will prevent brightness of regions that are not related to anatomy, that will potentially cause skewness or misleading in ehanncement, segmentation and intensity analysis.

![download (74)](https://github.com/user-attachments/assets/81455465-6e49-4a7d-8896-e01102a8efb0)

## Enhancement Concepts

This is the heart of my enhancement process for brain MRIs. MUSICA stands for Multiscale Image Contrast Amplification. It constructs a Gaussian pyramid. It downsamples by generating a sequence of increasing smaller images. After that, it upsamples by going backward with generating reconstructed images while amplifying the image details like edges, transitions, fine-grained artifacts that were previously lost between levels of downsampling. In other words, it emphasizes the detail by using an alpha multiplier and adds back into the new version with higher resolution at each scale. The results of this technique in my project is quite impressive and immensed when it can make fine brain structures like tissue and skull boundaries visually defined, which benefits manual interpretation and my later segmentation as well as classification.

1. Multiscale Image Enhancement (Laplacian Pyramid approach)

Decompose image into low- and high-frequency components (Gaussian + detail).

Amplify detail (high-frequency) layers.

Reconstruct the enhanced image via synthesis.

2. MUSICA (Multiscale Image Contrast Amplification)

Essentially: Apply the above with fine-tuned contrast amplification at each level.

These processe are crucial in enhancing image quality overall and promote finer details and objects. Without these processes, the original images' qualities are varied and undermined, which can cause unexpected outputs, namely in the segmented tumor for brain MRIs with tumor. The below is indentied with a mask on tumor but this mask also covers unwanted ares like normal soft tissue region.

![download (68)](https://github.com/user-attachments/assets/fdc95f37-6a1f-44f8-8457-7bd93dabce6d)

Now I will proceed with the MUSICA for MRIs with tumor(s).

![Cropped images](https://github.com/user-attachments/assets/ef1f6f52-51d8-434c-8200-fb23f6c381f2)

![download (17)](https://github.com/user-attachments/assets/63f1bc9f-5c3d-419c-98fc-0794edb9d5be)

![download (12)](https://github.com/user-attachments/assets/e389182f-c5e0-40e9-8431-8df5cdfd0507)

![download (11)](https://github.com/user-attachments/assets/0c65ccb8-08b0-4182-bb4f-6a1b9ed4947b)

![download (13)](https://github.com/user-attachments/assets/87a05431-6101-4578-bd02-3b967bf1b72d)


## K Means for Tumor Mask in colors

### Select a K based on basic domain-based approach (tumor, skull, brain tissue, background)

![download (14)](https://github.com/user-attachments/assets/5557cd08-41ed-4029-806b-9887c3ee306d)

![download (15)](https://github.com/user-attachments/assets/405ef50d-f565-4d3d-ad18-655efc59b36d)

![download (16)](https://github.com/user-attachments/assets/81394d7a-f58d-468c-9754-e7638bd0a569)

### Find the best K with revised clustering plots

For each image, I run KMeans clustering **for k values from 2 to 20**, and compute the following metrics for each k:

#### a. **WCSS (Within-Cluster Sum of Squares)**
- Measures how tightly packed the points are within each cluster.
- Lower values are better, but always decrease as k increases (so not ideal alone for choosing k).

#### b. **BCSS (Between-Cluster Sum of Squares)**
- Measures the separation between clusters.
- Higher values indicate better clustering.

#### c. **Silhouette Score**
- Measures how similar each point is to its own cluster vs. other clusters.
- Values range from -1 (poor) to +1 (ideal); higher values are better.

#### d. **Calinski-Harabasz Index**
- Compares cluster dispersion between and within clusters.
- Higher values indicate better clustering.

These metrics give a **balanced view of clustering performance**, considering both cohesion and separation.

![download (39)](https://github.com/user-attachments/assets/bd30daed-fc7c-47e6-bb03-6945921885d0)

Based on the **WCSS (Inertia)**, WCSS decreases significantly from k = 2 to k = 4, then tapers off, which is the elbow method suggesting k ≈ 4.

As per the **BCSS**, BCSS remains consistent increment with k, but the rate of increase slows after k = 4.

Regarding the **Silhouette Score**, the highest score at k = 2, but that is often too coarse. There is a noteworthy drop after k = 4, suggesting diminishing structure.

In terms of the **Calinski-Harabasz Index**, it increases linearly, which is useful but doesn't suggest the elbow like the WCSS or the Silhouette.

From this observation of all signals, the best should be **4**. which offers a balance of natural elbow in WCSS, a decent SIlhouette score and a fair trade-off of detail and interpretability for medical segmentation, such as: background, skull layer, brain tissue, tumor.

### Revise the MUSICA on non-flattened images before K Means and apply strict threshold for black pixels (enhance blackness for black/dark background)

![download (31)](https://github.com/user-attachments/assets/40dc64e1-68e2-44bf-9995-d2defb70f2cd)

![download (32)](https://github.com/user-attachments/assets/a8d28fab-e781-4dfb-ade4-2a26476c7bd1)

![download (33)](https://github.com/user-attachments/assets/9fdfb972-59fb-4795-b204-458edaac358a)

The surrounding background with some blue regions are now surpressed to become black before applying colorized K Means with suggested 4 colors.

### Find Better K based on new clustering plots with extended range up to 20

![download (45)](https://github.com/user-attachments/assets/080dbb69-fca8-4b53-bae9-c4dd8d66251a)

![download (46)](https://github.com/user-attachments/assets/9e14a174-b334-422f-bc58-f6f23689948b)

![download (47)](https://github.com/user-attachments/assets/a2cccff4-7148-4527-81cb-327fd3ccd2b1)

![download (48)](https://github.com/user-attachments/assets/332ccbbd-5b69-4f8b-b012-46610789fbda)

![download (49)](https://github.com/user-attachments/assets/ba20c226-ec46-4c68-b59e-75605d6f7263)

![download (50)](https://github.com/user-attachments/assets/9072aeea-9efe-4bd5-aaf2-3486cae01ddc)

![download (51)](https://github.com/user-attachments/assets/6c907a80-cae2-4cf6-8c95-96861bc5bfb3)

Based on the plots of 7 images, the best K is 9. This is a consistent balancing among high Silhouette Score, high Calinski-Harabasz Index, reasonable BCSS gain and WCSS drop-off.

### Colorized Clusters with K = 9

![download (52)](https://github.com/user-attachments/assets/bfa979b4-88aa-4a10-a106-146a87178900)

![download (53)](https://github.com/user-attachments/assets/c4d22175-0d2c-4587-9243-18a19c9465fe)

![download (54)](https://github.com/user-attachments/assets/7964aff1-81c1-436f-9983-6376b837b88b)


### Find Best K with Average Clustering Plots

I have designed this code to assess how well unsupervised KMeans clustering performs on the brain MRI tumor images that have been preprocessed with contrast enhancement. My goal is to pick some notable brain tumor images for the algorithm, find the best K by analyzing the clustering plots and assess how different values of k (number of clusters) affect clustering quality using multiple metrics and visualize the optimal k range. I decied to have the images clustered using KMeans for k from 2 to 20. Then I will evaluate by using WCSS, BCSS, Silhouette Score and Calinski-Harabasz Index on the charts, which will be plotted as averaged curves to help identify the best k for segmentation.

Before Clustering, I have to prepare the image data by converting them to grayscale to simplify the feature space and resize them to a uniform shape (128 × 128) to make sure of consistent input size. As reconmmended by Professor Stephen Elston, I also make sure to construct a feature matrix with 3 columns:

1. X position of the pixel
2. Y position of the pixel
3. Pixel intensity (from the grayscale image)

This combination of **spatial coordinates + intensity** helps KMeans segment not just based on brightness, but also on spatial proximity, which produce more insightful cluster shapes.

I picked these metrics for following reasons:

- WCSS (Within-Cluster Sum of Squares) assesses how tightly packed the points are within each cluster. So, in a case that lower values are better, it always decrease when k increases. At this point, I can see that this metric is not ideal alone for selecting k.

- BCSS (Between-Cluster Sum of Squares) assesses the separation between clusters. So, in a case of higher values, this means that there is better clustering.

- Silhouette Score assesses how similar each point is to its own cluster vs others with values ranging from -1, which is poor, to +1, which is good. Therefore, higher values indicate better clustering. This is going to be my top two metrics in determining k.

- Calinski-Harabasz Index commputes comparison of cluster dispersion between and within clusters. So, in case of higher values, there will be better clustering. This is going to be my top two metrics in selecting k as well.

A good advice from Professor Stephen Elston is that after picking some notable images with tumor as representatives from the datasets, I should average the clustering metrics across all of those images. This helps generalize my findings and provides a dataset-level view and assessment of clustering quality. After that, I will look at each plotto identify the optimal number of clusters when I can find the elbow point in WCSS, indicating where improvements start diminishing, the peak of Silhouette and Calinski-Harabasz indices and where BCSS growth starts to flatten.

At high level, the structure of my below code pipelines is as follows:

- Load 6 specific image files from the "yes" subfolder.

- Manually apply soft cropping on some images to isolate brain regions.

- Apply MUSICA (multiscale contrast enhancement) to each image.

- Convert images to grayscale and resize them to a standard size.

- Assess clustering quality for k = 2 to 20 using these metrics on plots:

1. WCSS (within-cluster sum of squares)

2. BCSS (between-cluster sum of squares)

3. Silhouette score

4. Calinski-Harabasz index

- Average these 4 scores across the 6 images.

- Visualize the average plots to guide the choice of optimal k.
  
As plotting the 4 clustering plots for each image would be tedious, I will average the 4 metrics for the chose images to provide 1 set of clustering plots of them only. Based on the new plots, the best K is now 10.

![download (69)](https://github.com/user-attachments/assets/2dad9dba-9fcd-4934-a494-1ab128755547)


### Colorized Clusters with Best K = 10

After identifying the best K, I will create a **segmentation mask** using clustering. First I will convert the MUSICA-enhanced image to grayscale and flattens it into a list of pixel intensity values. I will then run `KMeans` clustering with 10 clusters. Each cluster represents a region with similar intensity. After fitting, it assigns a cluster label to every pixel.

Then I will calculate the **average pixel intensity** for each cluster and sort them. The lowest-intensity cluster is assigned the color **black**, the highest gets **red**, and others get distinct colors in this intensity order:

> **Black → Purple → Blue → Cyan → Light Blue → Green → Yellow → Pink → Orange → Red**

The output is a **color-coded mask** where each region’s color corresponds to an intensity cluster. This step transforms raw grayscale intensity into **structured, interpretable regions**.

![download (71)](https://github.com/user-attachments/assets/c80ac188-738f-4cfd-8203-d42d6760bea3)

![download (70)](https://github.com/user-attachments/assets/d5d316f1-4540-4220-b0e7-b47c7f63c284)

![download (72)](https://github.com/user-attachments/assets/1c2573b6-9f67-47e6-aa29-e1fcc1a29a77)

The outputs are a color-coded mask where each region's color corresponds to an intensity cluster. This step transforms raw grayscale intensity into structured, interpretable regions. Based on the above results, I can evaluate that with k = 4, the clustering is overly simplified and too generalized so it becomes harder to seprate subtle tumor regions from brain tissue as it mixes complex patterns in brain tissues and tumor. When increasing k to 9, it starts to makes more reasonable separation of pixel intensity of brain tissues and tumor regions. However, it might still merge subtle sub-regions of tumor or tissues. When increasing k to 9, there are finer granularity as each region is better isolated, including the sub tumor variations. Still, this is my current hypothesis whether the colorized clustered images will be able to support the interpretability for model training and prediction later on. The backbone of the datastream for training is still the MUSICA-enhanced greyscale MRIs from with tumor and with tumors subfolders.

## Denoising Diffusion Implicit Models

This pipeline is a generative modeling approach that learns to generate realistic brain tumor MRI images by reversing a diffusion (noise) process. It has multiple real-world use cases in detection, segmentation and data augmentation.

| | |
| Goal | How This Pipeline Helps |
| 🧬 Tumor detection |	Data augmentation for CNN classifiers |
| 🎯 Segmentation | 	Preprocessing, denoising, and potential pretraining |
| 🔬 Research | Simulation of rare or hard-to-find tumor cases |
| 🛠️ Anomaly detection	| Measure how well a sample matches the learned tumor space |

```
Epoch 1/1000
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - i_loss: 2.4668 - n_loss: 0.7776 
```
![Epoch 1](https://github.com/user-attachments/assets/6b0fd6bd-460c-496e-ba21-bb308ade5dd0)

The generated images look like pure static/noise, which is as expected that the model hasn’t learned to denoise at all, proven by the image loss & noise loss are both high (~0.77 and ~2.4).

```
Epoch 300/1000
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - i_loss: 0.1418 - n_loss: 0.0835
```
![Epoch 300](https://github.com/user-attachments/assets/8a0b75cc-07ad-4ab0-b8b8-2998210b3463)

Tumor shapes begin to emerge, though textures are still noisy and blurry. The model has learned rough anatomical structure but not fine-grained features.

```
Epoch 600/1000
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 96ms/step - i_loss: 0.1301 - n_loss: 0.0631
```
![Epoch 600](https://github.com/user-attachments/assets/f98f9d57-53e5-4a7a-b5c0-d62d39465d96)

There are clearer boundaries and smoother tissue structure. The tumor regions are visually recognizable. Therefore, I can see that denoising is much more successful with noise loss drops to ~0.063.

```
Epoch 1000/1000
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 96ms/step - i_loss: 0.1161 - n_loss: 0.0601 
```
![Epoch 1000](https://github.com/user-attachments/assets/463eb9e4-73f2-4977-86c4-e1a369a84f74)

The images resemble actual MRIs with more kind of realistic tumor shapes, tissue contrast and fairly cleaner backgrounds.Textures are quite smoother and segmentation boundaries are quite sharper. Final losses show positive and promising signs with noise loss ~0.060 and image loss ~0.116. The model has converged and produces high-quality synthetic MRI tumor scans.

### Model Weights

![download (55)](https://github.com/user-attachments/assets/a0bee998-642c-4357-b384-f779b0163a9d)

Training and validation image loss (MAE between original and denoised images) both drop significantly. Training image loss reaches ~0.1; validation around ~0.2. Even though image loss isn't directly used to optimize the model, its decrease shows improved denoising quality. Low image loss implies the network is not only predicting noise well but is also reconstructing clean images that closely match the ground truth.

![download (57)](https://github.com/user-attachments/assets/831a26b5-d75f-48ff-84a1-4f9bbb090590)

Both curves show a sharp decrease in the first 100–200 epochs, then taper and stabilize. Validation loss is higher than training loss, but the gap is stable, which is not increasing significantly. The model is learning to predict the noise added during the forward diffusion process. A decreasing noise loss implies that the model is getting better at predicting noise accurately, which is critical for denoising in the reverse diffusion. The gap between training and validation suggests some generalization error, but not severe overfitting.

![download (56)](https://github.com/user-attachments/assets/1f90f651-bfb4-4baa-b109-f8de1e9dc4d5)

KID measures the distributional similarity between generated images and real images. The KID metric fluctuates heavily between epochs, staying roughly in the range of 1.66 to 1.76. The fluctuations suggest that while noise and image reconstruction improve, distribution matching is noisy, likely due to the given small validation set, high variance in InceptionV3 feature space and/or still imperfect denoising leading to unrealistic artifacts.

# Deep Learning Pipelines
---

## CLASSIFICATION

### ResNet50 (1st model)

ResNet50 is the first classification model in my project to detect the presence of brain tumors in MRI images. This is a binary classification task where Class 0 represents No Tumor and Class 1 represents Tumor Present. ResNet50 serves as a powerful feature extractor, especially when pretrained on ImageNet and works well on medical imaging tasks due to its benefits.

The first benefit comes from Transfer Learning that it leverages pretrained filters from large datasets like ImageNet to capture texture, shape and edges, even in grayscale-like MRI images. It also has a deep architecture where its residual blocks help train deeper networks without vanishing gradients. The ResNet also has proven performance as it is a standard benchmark in medical image classification and diagnostic imaging tasks. It is assumed to perform faster convergence when less data and training time are needed because initial layers already capture general features.  

In my pipeline, I will delope the 1st and 2nd ResNet50 Models with some differences between them.

This 1st model - Sequential API, uses a simpler, `Sequential` approach.

### 1. **Data Loading and Augmentation**

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalizes images using ImageNet mean/std
    validation_split=0.2,
    rotation_range=10, width_shift_range=0.05,
    height_shift_range=0.05, shear_range=0.05,
    zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
)
```

This version applies real-time augmentation and normalizes images to match ResNet50 expectations.

### 2. **Load Image Batches**

```python
train_generator = train_datagen.flow_from_directory(..., subset='training')
val_generator = train_datagen.flow_from_directory(..., subset='validation')
```

The code loads images and labels from folder structure using `class_mode='binary'`.

### 3. **Model Definition**

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False
```

Then I will load the ResNet50 without top classification layer. I will then freeze the convolutional base for transfer learning.

```python
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
```

I add a custom head on top with dense layers and dropout for regularization so my layers will end with sigmoid for binary classification.


### 4. **Compile and Train**

```python
lr_schedule = ExponentialDecay(1e-4, decay_steps=1000, decay_rate=0.9)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), binary_iou]
)
```

Here I will use exponential decay learning rate and binary IoU as a custom metric (same as the Segmenters) and train the model with Adam optimizer.

```python
model.fit(..., callbacks=[earlystop, checkpoint])
```

Here I will initiate training the model and save the best one based on validation AUC with Early Stop when the learning does not improve anymore to avoid overfitting.

![download (55)](https://github.com/user-attachments/assets/7ede18c8-7046-42d3-9e12-e57530028064)

![download (56)](https://github.com/user-attachments/assets/cd21b2cc-aa72-4d80-b7e1-dfc1cc5568ac)

### ResNet50 (2nd model)

My 2nd version - Functional API, is more modular, scalable and flexible for several reasons.

### 1. **Preprocessing**

```python
inputs = keras.Input(shape=img_size + (3,))
x = preprocess_input(inputs)
```

I will start by defining an explicit input layer and apply preprocessing inline using Keras Applications.

### 2. **ResNet50 as Backbone**

```python
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_tensor=x,
    pooling='avg'
)
base_model.trainable = False
```

My ResNet50 is integrated and serves as a feature extractor with average pooling included directly.

### 3. **Custom Classification Head**

```python
x = layers.Dense(1024, activation='relu')(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
```

Here I will configure a deeper fully connected head with progressive dimensionality reduction and regularization.

```python
model = keras.Model(inputs, outputs)
```

Then I will wrap the entire flow from input to base\_model and then head into a single model object.

### 4. **Compile and Train**

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), binary_iou]
)
```

As advised, I will add weight decay to reduce overfitting, and the same metrics.

```python
model.fit(..., callbacks=[earlystop, checkpoint])
```

I also use early stopping and checkpointing just like before to save compute and avoid over fitting.

### 5. **Fine-Tuning Phase**

```python
base_model.trainable = True  # Unfreeze
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    ...
)
model.fit(...)
```

After training the top, I will unfreeze the base model and fine-tune the entire network with a slower learning rate.

---

Regarding the code differences:

| Section                | 1st ResNet50             | 2nd ResNet50                          |
| ---------------------- | ------------------------ | ------------------------------------- |
| API                    | `Sequential()`           | `Functional()`                        |
| Input Size             | `(224, 224, 3)`          | `(300, 300, 3)`                       |
| Preprocessing          | Via ImageDataGenerator   | Inline in model graph                 |
| Base Model Pooling     | Manual `GlobalAvgPool2D` | `pooling='avg'` argument              |
| Classification Head    | Dense(512 → 128 → 1)     | Dense(1024 → 512 → 128 → 1)           |
| Regularization         | Dropout only             | Dropout + BatchNorm after every Dense |
| Learning Rate Schedule | `ExponentialDecay(1e-4)` | `ExponentialDecay(1e-6)`              |
| Fine-Tuning            | Optional                 | Fully integrated with lower LR        |

---

Regarding the feature differences:

| Feature / Setting       | **1st ResNet50 Model** (Sequential)                     | **2nd ResNet50 Model** (Functional API)                      |
| ----------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **Architecture Type**   | `Sequential` API                                        | `Functional` API                                             |
| **Input Preprocessing** | Done via `ImageDataGenerator(preprocessing_function)`   | Done inline with `preprocess_input(inputs)`                  |
| **Base Model Output**   | No `pooling` argument (manual `GlobalAveragePooling2D`) | `pooling='avg'` used in ResNet50 initializer                 |
| **Intermediate Layers** | Fewer: Dense(512 → 128)                                 | Deeper: Dense(1024 → 512 → 128) + more BatchNorm + Dropout   |
| **Model Flexibility**   | Less flexible (e.g., hard to use multi-input models)    | More modular, extendable (e.g., attention, branching, etc.)  |
| **Weight Decay**        | Only applied in optimizer during fine-tuning            | Applied in both initial and fine-tuning phases               |
| **Training Strategy**   | One-pass training + optional fine-tune                  | Clear two-phase strategy: frozen base → fine-tune full model |
| **Input Shape**         | 224×224 (lighter on memory)                             | 300×300 (higher resolution = richer features)                |

---

In summary, the 1st ResNet50 is a better choice for a quick prototype as it has a simple `Sequential` setup whle the 2nd model is not ideal for large-scale tuning. In terms of performance-focused target, the 1st has less depth and felxibility hwile the 2nd was designed with better architecture and resolution. Both has good Transfer learning as the 1st leverages pretrained layers and the 2nd has it with deeper fully connected layers. However, the 1st is limited in terms of fine-tuning flexibility. Meanwhile, the 2nd has this privilege with a well=plannned unfreezing backbone. Therefore, the 2nd model is configured to be the improved and production-worth version of the 1st by having better layer structure that handles higher-resolution inputs and uses the API for flexibility. Lastly, the 2nd model adopts a two-phase training schedule that aligns with the best prices in deep learning.

The ResNet50 model achieves very low training loss and high accuracy quickly but fails to improve validation metrics after 15–20 epochs. Also, validation loss increases, indicating memorization over learning, probably because of high model capacity, no regularization like dropout, weight decay and too gressive learning rate as well as lack of data and poor data augmentation at this stage.

![Resnet with 100 epochs plot 1](https://github.com/user-attachments/assets/c78a1b07-2853-4e6e-b6a5-66324afa7f45)

![Resnet with 100 epochs plot 2](https://github.com/user-attachments/assets/5b74b586-cc09-419b-85a5-f2331cdc2011)

### EfficientNetV2L (1st model)

In my project, classification using pretrained models like ResNet50 or EfficientNetV2L is essential because pretrained CNNs (like EfficientNetV2L) provide me with strong visual feature extractors pretrained on ImageNet, allowing faster convergence and better generalization on small medical datasets that I have them limited and have to augment to enrigh them. Another model like EFficientNetV2L reduces my need for building and training deep architectures from scratch, especially important for medical imaging, where labeled data in my dataset is limited.

There are some key differences betwen my 2 models but they are essential:

| Feature                   | Model 1 (Fast LR)                               | Model 2 (Slow LR)                                     |
| ------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| **Initial Learning Rate** | `1e-4`                                          | `1e-5` (slower and more stable)                       |
| **Decay Rate**            | `0.9`                                           | `0.85`                                                |
| **Decay Speed**           | Faster (fewer steps to decay)                   | Slower decay (more gradual fine-tuning)               |
| **Early Stopping**        | `EarlyStopping` applied                         | `No EarlyStopping` explicitly shown in fine-tuning    |
| **Fine-tuning Phase**     | Fine-tune at `1e-5`                             | Fine-tune at `1e-6` (more cautious learning)          |
| **Output Model File**     | `efficientnetv2l_binary_model.keras`            | `efficientnetv2l_binary_model_v2.keras`               |
| **Code Structure**        | Single script with early stopping + fine-tuning | More modular which shows clear split between phase 1 and phase 2 |

---

Regarding the comparison between them in code, there are shared points between both:

* **Architecture**: EfficientNetV2L without top, followed by:

  * GlobalAveragePooling → BatchNorm → Dropout → Dense(512) → Dropout → Dense(256) → Dropout → Dense(1 sigmoid)
* **Loss**: `binary_crossentropy`
* **Metrics**: `accuracy`, `AUC`, and custom `binary_iou`
* **Data**: `ImageDataGenerator` with the same augmentation, using the same dataset path.

But there are some actual differences between them in codes, especially about learning pace:

**1st Model (Fast LR)**:

```python
lr_schedule = ExponentialDecay(1e-4, decay_steps=1000, decay_rate=0.9)
...
model.fit(..., epochs=30, callbacks=[earlystop])
model.compile(..., learning_rate=1e-5)
model.fit(..., epochs=10)
```

**2nd Model (Slow LR)**:

```python
lr_schedule = ExponentialDecay(1e-5, decay_steps=1000, decay_rate=0.85)
...
model.fit(..., epochs=30)
model.compile(..., learning_rate=1e-6)
model.fit(..., epochs=10)
```

These differences matter as lower initial LR and slower decay in the 2nd model make it more robust against overfitting and noisy gradient updates. I will consider this as suitable for sensitive medical classification tasks. A noteworthy point is that my 1st model has faster training but risks overshooting minima, while my 2nd model has a more careful convergence so it trades time for precision. Lastly, my 1st model explicit uses `EarlyStopping` to save training time by stopping early if no improvement is observed.

![download (57)](https://github.com/user-attachments/assets/0ba13af8-ed10-4a1c-ad28-a9493592f8bd)

![download (58)](https://github.com/user-attachments/assets/c2de14dd-e025-40f0-991f-7dcc3946a85b)


![download (59)](https://github.com/user-attachments/assets/df67b90e-c89f-4cca-a534-41b6d4e962ce)

![download (73)](https://github.com/user-attachments/assets/aeaf7a70-53d1-474f-aaeb-1cfd1b4a66a2)


### ConvNeXt Base (1st model)

I also chose ConvNeXt in my Classification pipeline as it is a modernized convolutional neural network (CNN) that matches the performance of Vision Transformers (ViTs) while keeping the efficiency of CNNs. There are certain benefits of developing this model. The first benefit also comes from the ImageNet pre-trained for strong transfer learning. Secondly, it is inspired by ViTs but faster and easier to train and it can work well on small-to-medium medical datasets. Furthermore, the ConvNeXtBase provides state-of-the-art accuracy and handles fine textures and localization well, which I find it useful for subtle tumor features. Therefore, it is fficient for both training and inference in my medical workflows.

There are several differences my models in 2 versions:

| Feature                       | **Model 1** (ConvNeXt v1)                                     | **Model 2** (ConvNeXt v2)                                            |
| ----------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Data loader**               | `ImageDataGenerator.flow_from_directory`                      | `image_dataset_from_directory` (native tf.data API)                  |
| **Preprocessing**             | Preprocessing inside `ImageDataGenerator`                     | Applied explicitly using `convnext.preprocess_input(inputs)`         |
| **Batch performance**         | Slower (uses legacy Keras generator)                          | Faster (TensorFlow-native pipeline with prefetching)                 |
| **Initial learning rate**     | `1e-4`                                                        | `1e-5`                                                               |
| **Decay rate**                | 0.9                                                           | 0.85                                                                 |
| **EarlyStopping**             | Yes                                                           | No callbacks specified during training                               |
| **Fine-tuning learning rate** | `1e-5`                                                        | `1e-6`                                                               |
| **Model save**                | Not saved                                                     | Saved at end using `model.save(model_path)`                          |
| **Model structure**           | Same architecture: ConvNeXtBase → GAP → Dense Layers → Output | Same                                                                 |
| **Dataset interface**         | Keras legacy generator (shuffle=True / False manually)        | `tf.data.Dataset` with `AUTOTUNE` for efficient batching + shuffling |

Some other aspects I can mention about the similarities and differences between the 2 models are:

 **Data Loaders**

Model 1: `ImageDataGenerator` (Keras legacy)

```python
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.convnext.preprocess_input,
    validation_split=0.2
)
```
This generator applies preprocessing and data augmentation and uses ConvNeXt-specific normalization like rescaling, mean/std adjust.

```python
train_ds = datagen.flow_from_directory(..., subset='training')
val_ds = datagen.flow_from_directory(..., subset='validation')
```

The 1st model loads images from folders with `"tumor"` and `"no_tumor"` subdirectories and automatically assigns binary labels based on folder names.

Meanwhile the 2nd model approaches a bit differently with `image_dataset_from_directory` (TF native).

```python
train_ds = keras.utils.image_dataset_from_directory(
    ..., validation_split=0.2, subset="training", label_mode="binary"
)
val_ds = keras.utils.image_dataset_from_directory(... subset="validation")
```

In this way, the 2nd model is easier to scale and faster on GPU and explicitly sets image size and labels. Its `prefetch(AUTOTUNE)` can improve performance by overlapping CPU/GPU work.

There is a same approach of both models leveraging **Learning Rate Schedule**:

```python
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4 or 1e-5,
    decay_steps=1000,
    decay_rate=0.9 or 0.85
)
```

This implements:

$$
lr_t = lr_0 \cdot decay\_rate^{t / decay\_steps}
$$


So, basically, the learning starts with small LR to avoid catastrophic forgetting and decays it slowly over time.

There is a similar model construction with my ConvNeXtBase:

```python
base_model = keras.applications.ConvNeXtBase(
    include_top=False,
    weights='imagenet',
    input_shape=img_size + (3,),
    pooling='avg'
)
base_model.trainable = False  # Freeze during initial training
```

Here I load pretrained ConvNeXtBase without final classification head and use global average pooling (GAP) to convert 2D features to 1D.

```python
inputs = keras.Input(shape=img_size + (3,))
x = keras.applications.convnext.preprocess_input(inputs)
```

Then I will wrap input image and apply ConvNeXt-specific normalization.

```python
x = base_model(x, training=False)
x = layers.Dense(1024, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```

At this stage, my Fully connected (FC) layers can refine features:

  - `Dense(1024 → 512 → 128)`
  - BatchNorm stabilizes
  - Dropout avoids overfitting
  - `Dense(1, sigmoid)` for binary classification output

```python
model = keras.Model(inputs, outputs)
```

The compilation is similar to the other models I developed before with an optimizer, accuracy, AUC and IoU.

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name='auc'), binary_iou]
)
```

- `Adam` for adaptive optimization
- `binary_crossentropy` for binary output
- `AUC` for class separability
- `binary_iou` for localization

When it comes to training,

```python
earlystop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[earlystop])
```

I will train for max 30 epochs and apply Early stopping if validation AUC doesn't improve for 5 epochs.

The model itself also needs a Fine-Tuning Phase:

```python
base_model.trainable = True  # Unfreeze ConvNeXt
model.compile(... learning_rate=1e-5 or 1e-6)
model.fit(train_ds, validation_data=val_ds, epochs=10–30)
```

My intention is to allow backpropagation through the pretrained ConvNeXt and use very low LR to avoid damaging pretrained weights.

In summary, based on the configurations, I can use the 1st model for more controlled experiments, especially if I want early stopping. Otherwise, I can use the 2nd model for production pipelines with better performance and easier scaling with distributed datasets.

![download (74)](https://github.com/user-attachments/assets/c6b7dc92-60da-4a57-a679-ce4fbeb59bc9)

### ConvNeXt Base (2nd model)

![Convnext with 100 epochs](https://github.com/user-attachments/assets/0a89e2f9-7a3e-4b07-861f-e016febf92c8)

The ConvNeXtBase experiences that the Train Accuracy is lightly lower but stable (~98–99%) and the Validation Accuracy is more stable at high 92–94% range while the Train Loss shows gradual and smoother decrease, and the Validation Loss illustrates a smooth trend with lower overfitting symptoms. These behaviors indicate a better generalization and less variance, therefore, the risk of overfitting is moderate to low as the Convergence Stability is evidenced with more consistent learning. However, before optimizing the model, I will need to apply these strategies for all models such as:

- Data Augmentation simulates larger dataset and adds regularization.
  
- EarlyStopping	prevents unnecessary epochs once validation loss or accuracy stagnates.
  
- Learning Rate Scheduler	assists model fine-tune without overshooting minima, such as Exponential Decay.

- Freezing Base Layers avoids destroying pretrained features too early.
  
- Reducing Learning Rate helps especially in the second half of training.

### ConvNeXt Base (after Augmentation, slow Learning Rate & more Weight Decay)

![download (22)](https://github.com/user-attachments/assets/01e3da85-52f0-4213-a61c-daf5e090c389)

![download (9)](https://github.com/user-attachments/assets/e7726778-a60e-4fd2-a7f7-e9d9fcaa6890)

### Comparison of Classifiers

Here’s a detailed interpretation of the comparison:

```
| Model                | Val Accuracy | Train F1 | Val F1  | Train IoU | Val IoU | AUC (Val) | Notes                                         |
|----------------------|--------------|----------|---------|-----------|---------|-----------|-----------------------------------------------|
| ConvNeXtBase (2nd)   | 99.97%       | 0.9996   | 0.9996  | 0.9992    | 1.0000  | 1.0000    | Best metrics overall, high precision & IoU    |
| EfficientNetV2L (1st)| 98.43%       | 0.9984   | 0.9900  | 0.9990    | ~0.99   | 0.9985    | Achieved great balance, high tumor precision  |
| ResNet50 (2nd)       | 96.87%       | 0.9914   | 0.9687  | 0.9994    | ~0.84   | 0.9937    | Reached stronger result after fine-tuning     |
| ConvNeXtBase (1st)   | 96.80%       | 1.0000   | 0.9682  | 1.0000    | ~0.93   | 0.9963    | High accuracy, less stable IoU on val         |
| EfficientNetV2L (2nd)| 97.77%       | 0.9996   | 0.9777  | 0.9992    | ~0.96   | 0.9972    | Great after fine-tuning, slow start           |
| ResNet50 (1st)       | 96.00%       | 0.9942   | 0.9445  | 0.9999    | ~0.91   | 0.9858    | Earlier convergence, some instability         |
```

Overall, the best performer is the **2nd ConvNeXtBase** that reached the highest **Validation Accuracy**: 0.9997, a perfect discrimination shown by **AUC**: 1.0000, the most ideal threshold overlap by **IoU (Validation)**: 1.0000, and all ultimate **Precision / Recall / F1 (Tumor)**: 1.00. As a result, my **2nd ConvNeXtBase** model almost perfectly classifies tumor presence and absence. The exceptional IoU at 1.0 confirms its high overlap between prediction and ground truth. However, furtehr process should be to verify overfitting risk as all signs are too perfect.

The second best performer is the **1st EfficientNetV2L** with **Val Accuracy**: 0.9843, **AUC**: 0.9985, **Tumor Precision, Recall, F1** at 1.00, 0.97, 0.99, respectively. Therefore, my **1st EfficientNetV2L** is very high-performing and balanced with a slightly lower recall, 3% miss rate on tumors, but excellent precision and generalization.

The third best performer is the **2nd ResNet50** which is better than the 1st ResNet50 across all metrics. It shows good stability after fine-tuning.

The fourth runner is the **2nd EfficientNetV2L** which illustrates slightly lower than its first version, probably due to slower convergence or insufficient fine-tuning.

The fifth ranking belongs to the **1st ResNet50** which has 
lower recall and unstable IoU. Also, earlier convergence led to overconfidence without generalization.

The final place is the **1st ConvNeXtBase** which sees great AUC but moderate recall at 0.94. Still, it is a solid all-around choice.

## SEGMENTATION

### Segment Anything Model with Transformers (SAM)

SAM has the following components:

| ![](https://imgur.com/oLfdwuB.png) |
|:--:|
| Image taken from the official [SAM blog post](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) |

The image encoder is responsible for computing image embeddings. When interacting with
SAM, we compute the image embedding one time (as the image encoder is heavy) and then
reuse it with different prompts mentioned above (points, bounding boxes, masks).

Points and boxes (so-called sparse prompts) go through a lightweight prompt encoder,
while masks (dense prompts) go through a convolutional layer. We couple the image
embedding extracted from the image encoder and the prompt embedding and both go to a
lightweight mask decoder. The decoder is responsible for predicting the mask.

| ![](https://i.imgur.com/QQ9Ts5T.png) |
|:--:|
| Figure taken from the [SAM paper](https://arxiv.org/abs/2304.02643) |

SAM was pre-trained to predict a _valid_ mask for any acceptable prompt. This requirement allows SAM to output a valid mask even when the prompt is ambiguous to understand -- this
makes SAM ambiguity-aware. Moreover, SAM predicts multiple masks for a single prompt.

I will check out the [SAM paper](https://arxiv.org/abs/2304.02643) and the [blog post](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) to learn more about the additional details of SAM and the dataset used to pre-trained it.

## Running inference with SAM

There are three checkpoints for SAM:

* [sam-vit-base](https://huggingface.co/facebook/sam-vit-base)
* [sam-vit-large](https://huggingface.co/facebook/sam-vit-large)
* [sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge).

I load `sam-vit-base` in
[`TFSamModel`](https://huggingface.co/docs/transformers/main/model_doc/sam#transformers.TFSamModel).

Then, I also need `SamProcessor`for the associated checkpoint. Let's define a set of points I will use as the prompt.

![download (58)](https://github.com/user-attachments/assets/f98dbd85-1e1d-4550-937e-c282229c8a6f)

`outputs` has got two attributes of our interest:

* `outputs.pred_masks`: which denotes the predicted masks.
  
* `outputs.iou_scores`: which denotes the IoU scores associated with the masks.
  
Let's post-process the masks and visualize them with their IoU scores:

![download (59)](https://github.com/user-attachments/assets/64cd1c3a-85ac-43f8-ac4d-f2e48a82ab47)

As can be noticed, all the masks are _valid_ masks for the point prompt I provided.

### Extract the predicted mask into Ground Truth Image for later Segmentation Pipeline with DeepLab V3+ (to be continued)

![download (77)](https://github.com/user-attachments/assets/716dda9b-637e-4ca2-bb2f-61fc4d00e39f)

![download (76)](https://github.com/user-attachments/assets/9af4375e-a23d-448b-ae15-0a3f904640a6)

![download (78)](https://github.com/user-attachments/assets/7ea9e5e5-c914-4a7a-810d-5659472c86d4)


### Apply SAM's points on MRIs with tumors

![download (80)](https://github.com/user-attachments/assets/bc3a4566-e933-4a70-853f-077e789e132b)

After supervising the SAM's application of pinpointing the tumors on the original images, I will show the masks on those tumors to observe the SAM's predicted masks.

### Preparing Images in Tensorflow array as original and masked images for Segmentation Pipeline:

![download (84)](https://github.com/user-attachments/assets/6670fc34-92cc-45fa-a2a8-52b76a774806)

![download (85)](https://github.com/user-attachments/assets/ceb9e870-b348-4210-8c13-641ab2cfe7d4)

![download (86)](https://github.com/user-attachments/assets/906c1ed0-8265-4085-8f50-330a84ac7554)

### DeepLab V3+ Diagram

![deeplabv3_plus_diagram mask](https://github.com/user-attachments/assets/1fd879c8-5b20-474d-a396-5d9c0af46665)

I also chose ConvNeXt in my Classification pipeline as it is a modernized convolutional neural network (CNN) that matches the performance of Vision Transformers (ViTs) while keeping the efficiency of CNNs. There are certain benefits of developing this model. The first benefit also comes from the ImageNet pre-trained for strong transfer learning. Secondly, it is inspired by ViTs but faster and easier to train and it can work well on small-to-medium medical datasets. Furthermore, the ConvNeXtBase provides state-of-the-art accuracy and handles fine textures and localization well, which I find it useful for subtle tumor features. Therefore, it is fficient for both training and inference in my medical workflows.

There are several differences my models in 2 versions:

| Feature                       | **Model 1** (ConvNeXt v1)                                     | **Model 2** (ConvNeXt v2)                                            |
| ----------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Data loader**               | `ImageDataGenerator.flow_from_directory`                      | `image_dataset_from_directory` (native tf.data API)                  |
| **Preprocessing**             | Preprocessing inside `ImageDataGenerator`                     | Applied explicitly using `convnext.preprocess_input(inputs)`         |
| **Batch performance**         | Slower (uses legacy Keras generator)                          | Faster (TensorFlow-native pipeline with prefetching)                 |
| **Initial learning rate**     | `1e-4`                                                        | `1e-5`                                                               |
| **Decay rate**                | 0.9                                                           | 0.85                                                                 |
| **EarlyStopping**             | Yes                                                           | No callbacks specified during training                               |
| **Fine-tuning learning rate** | `1e-5`                                                        | `1e-6`                                                               |
| **Model save**                | Not saved                                                     | Saved at end using `model.save(model_path)`                          |
| **Model structure**           | Same architecture: ConvNeXtBase → GAP → Dense Layers → Output | Same                                                                 |
| **Dataset interface**         | Keras legacy generator (shuffle=True / False manually)        | `tf.data.Dataset` with `AUTOTUNE` for efficient batching + shuffling |

Some other aspects I can mention about the similarities and differences between the 2 models are:

 **Data Loaders**

Model 1: `ImageDataGenerator` (Keras legacy)

```python
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.convnext.preprocess_input,
    validation_split=0.2
)
```
This generator applies preprocessing and data augmentation and uses ConvNeXt-specific normalization like rescaling, mean/std adjust.

```python
train_ds = datagen.flow_from_directory(..., subset='training')
val_ds = datagen.flow_from_directory(..., subset='validation')
```

The 1st model loads images from folders with `"tumor"` and `"no_tumor"` subdirectories and automatically assigns binary labels based on folder names.

Meanwhile the 2nd model approaches a bit differently with `image_dataset_from_directory` (TF native).

```python
train_ds = keras.utils.image_dataset_from_directory(
    ..., validation_split=0.2, subset="training", label_mode="binary"
)
val_ds = keras.utils.image_dataset_from_directory(... subset="validation")
```

In this way, the 2nd model is easier to scale and faster on GPU and explicitly sets image size and labels. Its `prefetch(AUTOTUNE)` can improve performance by overlapping CPU/GPU work.

There is a same approach of both models leveraging **Learning Rate Schedule**:

```python
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-4 or 1e-5,
    decay_steps=1000,
    decay_rate=0.9 or 0.85
)
```

This implements:

$$
lr_t = lr_0 \cdot decay\_rate^{t / decay\_steps}
$$


So, basically, the learning starts with small LR to avoid catastrophic forgetting and decays it slowly over time.

There is a similar model construction with my ConvNeXtBase:

```python
base_model = keras.applications.ConvNeXtBase(
    include_top=False,
    weights='imagenet',
    input_shape=img_size + (3,),
    pooling='avg'
)
base_model.trainable = False  # Freeze during initial training
```

Here I load pretrained ConvNeXtBase without final classification head and use global average pooling (GAP) to convert 2D features to 1D.

```python
inputs = keras.Input(shape=img_size + (3,))
x = keras.applications.convnext.preprocess_input(inputs)
```

Then I will wrap input image and apply ConvNeXt-specific normalization.

```python
x = base_model(x, training=False)
x = layers.Dense(1024, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
```

At this stage, my Fully connected (FC) layers can refine features:

  - `Dense(1024 → 512 → 128)`
  - BatchNorm stabilizes
  - Dropout avoids overfitting
  - `Dense(1, sigmoid)` for binary classification output

```python
model = keras.Model(inputs, outputs)
```

The compilation is similar to the other models I developed before with an optimizer, accuracy, AUC and IoU.

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name='auc'), binary_iou]
)
```

- `Adam` for adaptive optimization
- `binary_crossentropy` for binary output
- `AUC` for class separability
- `binary_iou` for localization

When it comes to training,

```python
earlystop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[earlystop])
```

I will train for max 30 epochs and apply Early stopping if validation AUC doesn't improve for 5 epochs.

The model itself also needs a Fine-Tuning Phase:

```python
base_model.trainable = True  # Unfreeze ConvNeXt
model.compile(... learning_rate=1e-5 or 1e-6)
model.fit(train_ds, validation_data=val_ds, epochs=10–30)
```

My intention is to allow backpropagation through the pretrained ConvNeXt and use very low LR to avoid damaging pretrained weights.

In summary, based on the configurations, I can use the 1st model for more controlled experiments, especially if I want early stopping. Otherwise, I can use the 2nd model for production pipelines with better performance and easier scaling with distributed datasets.


### DeepLab V3+ Results with normal learning rate

![download (24)](https://github.com/user-attachments/assets/3b13e24e-9d25-450d-b66a-126aa3b7cd41)

![download (23)](https://github.com/user-attachments/assets/4fbc060e-96e9-4e6d-9412-8e126e6b45a5)

### DeepLab V3+ Results with slow learning rate and low weight decay to optimize efficiency

After implementing the original DeepLabV3+ pipeline, I will customize the pipeline by integrating learning rate scheduling and regularization techniques for better generalization. A table of comparison between my original and updated implementations will be more visibly understandable.

| Component                                               | Original Version                              | Updated Version                                          | Impact & Reasoning                                                                                                            |
| ------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Optimizer                                           | `Adam(learning_rate=1e-4)`                        | `Adam(learning_rate=ExponentialDecay, weight_decay=0.01)`    | Generalize better by having dynamic learning rate and L2 regularization so training can be stabilized by slowing learning gradually. |
| Learning Rate Schedule                              | Constant at `1e-4`                            | `ExponentialDecay(initial=1e-4, decay_rate=0.9, steps=1000)` | Learning rate is reduced progressively, leaving more chance for better convergence over training timeline, minimizing risks of shootups.        |
| Weight Decay (L2)                                   | Not included                                    | `weight_decay=0.01` in Adam optimizer                      | Penalize large weights to fight against overfitting.                |
| Epochs                                              | `epochs=10`                                       | `epochs=100`                                                 | Undergo longer training schedule with smaller steps to refine masks better for subtle tumor boundaries.          |
| EarlyStopping                                       | Not used                                        | `EarlyStopping(patience=5)`                                | Training can be auto-stopped when validation loss stops improving to minimize overfitting and save computation costs                          |
| Model architecture (ASPP, ResNet skip, conv blocks) | Same                                            | Same                                                       | My intention is for hyperparameter/training-level upgrade, not for model design.                              |
| Metrics                                             | `["accuracy", iou_metric]`                        | Same                                                         | This is one of the most meaningful metric for segmentation where tumor region is way smaller than background.                             |
| Loss                                                | `SparseCategoricalCrossentropy(from_logits=True)` | Same                                                         | Remain this option choice as my mask labels are integers (0 or 1).                                         |

Some noteworthy points about the upgrades are as below:

1. ExponentialDecay:

It starts with a higher learning rate to escape poor local minima and decays gradually to fine-tune weights.

$$
lr_t = lr_0 \cdot \text{decay}_{\text{rate}}^{\frac{\text{step}}{\text{decay}_{\text{steps}}}}
$$


2. Weight Decay (L2 Regularization):

It adds a penalty to large weight values in my loss function. This is proven effective in having model complexity reduced. I think this is helpful for noisy or small and high-stake datasets like my current medical scans.

3. EarlyStopping:

It helps to avoid wasting time and computation costs on further training as soon as model performance plateaus on validation data. Therefore, it can avoid overfitting beyond the best epoch.

4. Extended Epochs with 100 epochs:

The original DeepLabV3+ is a heavy model so starting off with 10 epochs might be too few to converge meaningfully enough. So I increase the epoch up to 100 epochs to generalize better learning, especially as decay slows learning over time.

Here are some of my upgrades on the training parameters compared to both the original and first improved DeepLabV3+ versions.

| Aspect                               | Earlier Improved Version                              | New Version | Reasoning                                                                                                |
| ------------------------------------ | --------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Initial Learning Rate            | `1e-4`                                                    | `1e-5`          | Undergo slower start for finer gradient updates, which is beneficial for noisy and complex medical images and avoids overshooting. However, a risk of longer convergence is imminent. |
| Decay Rate                       | `0.9`                                                     | `0.99`          | Undergo slower decay and reduces learning rate gradually. This makes the model learn longer at a decent oace for more epochs.                                 |
| Weight Decay (L2 Regularization) | `0.01`                                                    | `0.001`         | Less This is a more aggressive regularization to make more model still capable of fitting the data and avoid underfitting complex features like irregular tumor shapes.  |

Overall, these improvements are considereably good as my model was previously underfitting. And I have enough GPU time for longer training on my complex data,
which requires more subtle pattern recognition.

![download - 2025-04-24T232702 105](https://github.com/user-attachments/assets/142159af-a3bf-4b30-b174-b15e518c4308)

![download - 2025-04-24T232654 103](https://github.com/user-attachments/assets/89ab7a0f-643e-4b88-9ec8-4f7af40e913a)

![download (100)](https://github.com/user-attachments/assets/f901190c-7bcb-4a59-be25-491cf75f44c1)

![download (99)](https://github.com/user-attachments/assets/0853ff46-afbd-461c-a712-4afed659d40b)

![download - 2025-04-24T234231 574](https://github.com/user-attachments/assets/ebef64ac-6560-4a9f-8acf-96c6fb65aad8)

Looking at the metric performance, the model I have developed has very high accuracy at ~99.8%, steady IoU improvement at ~97.2%, low and consistent training & validation loss, relatively stable validation performance, although it iss starting to plateau around epoch 8–11. When reviewing the Plots, the Loss sees a slight upward trend in val_loss, which is possible early signs of overfitting. The Accuracy shows very high and stable rates so there is no collapse in performance so far. The IoU shows slight fuctuation but itend to be good generally, showcasing that th emodel is segmenting well. However, I will still need to tackle the imbalance, overfitting risks and lack of data training.

## SegFormer:

Next, I will implement **custom SegFormer model** for **brain tumor segmentation**. This will be an advanced choice complementing DeepLabV3+ for this purpose. I would not say this is a choice improving upon DeepLabV3+ as I will compare the performance later.

There are different reasons why I decided to move forward woth SegFormer for Medical Image Segmentation. First, SegFormer offers state-of-the-art accuracy when it blends Transformers with CNN-like efficiency, excelling at fine-grained boundary localization where small details and large context can be captured. Secondly, the transformer blocks in the SegFormer allow global context modeling across image patches. This is different from CNNs that has heavy reliance on receptive fields. SegFormer uses a simple, lightweight MLP-based decoder. This is crucial as less computational costs are consumed while maintaining its performance at  high accuracy. Also, SegFormer offers multi-scale feature fusion where I can upsample and concatenate feature maps from 4 different spatial scales. These behaviors are similar to UNet or DeepLab skip connections but they can be more efficiently. Lastly, SegFormer tends to be more robust to class imbalance with global reasoning via attention can have small tumor regions detected, which might be ignored by CNNs.

Regarding the Data Input and Preprocessing, there is a similar pipeline as DeepLabV3+ where I loads and resize images to `(512, 512)` and have masks  binarized with `0 = background` and `1 = tumor`.

The next step in the DeepLabV3+ is about Patch Extraction and Encoding Layers** which flattens image into tokens by `tf.image.extract_patches` and have positional info encoded when it is lost during patching. My SegFormer pipeline simplfies this so the pipeline skips this and replaces it with progressive CNN downsampling. This downsampling will mimic the processes of extracting hierarchical token and conducting spatial reduction. In fact, these 4 downsampling stages are the CNN backbone as below, corresponding to features at different resolutions.

- s1: 4× downsampled
- s2: 8×
- s3: 16×
- s4: 32×

The backbone of my SegFormer is a pure CNN and transformer-style feature aggregation while the DeepLabV3+ backbone is a ResNet50 and ASPP (dilated convolutions).

After the downsampling, here comes the MLP decoder with upsampling to replace heavy transposed convolutions used in the prior DeepLabV3+ and later UNet. Each stage from `s1` to `s4` is passed through a `Conv2D(128, 1)` to have a unification in depth. Therefore, any features will be resized to the unified `32×32` resolution. Actually, They are concatenated and passed through two layers of `Conv2D(256, 3x3)`, a final `Upsampling(16x)` to upsample from `32x32` to `512x512` and then reach a `Conv2D(num_classes, 1)`, which is output logits per pixel.

Regarding the Training elements, both models remain the same in using the Optimizer `Adam(1e-4)`, Loss `SparseCategoricalCrossentropy(from_logits=True)`, accuracy metric and epoch number by 10.


### Comparison of Segmenters

Here is a detailed comparison table of the five segmentation models I have tested:

```markdown
| Model               | Val Accuracy | Train F1 | Val F1  | Train IoU | Val IoU | AUC (Val) | Notes                                      |
|---------------------|--------------|----------|---------|-----------|---------|-----------|--------------------------------------------|
| DeepLabV3+ (2nd)    | 99.72%       | 0.9818   | 0.9776  | 0.9704    | 0.9578  | 0.9998    | Best overall metrics, stable validation    |
| DeepLabV3+ (1st)    | 99.65%       | 0.9743   | 0.9724  | 0.9537    | 0.9474  | 0.9998    | Close 2nd, faster to train (10 epochs)     |
| DeepLabV3+ (3rd)    | 99.57%       | 0.9715   | 0.9662  | 0.9585    | 0.9342  | 0.9997    | Solid, slightly less generalizable         |
| Custom U-Net        | 99.58%       | 0.9708   | 0.9667  | (est. ~0.95)| ~0.93 | 0.9997    | Lightweight, good performance              |
| Xception-style U-Net| 99.02%       | 0.9400   | 0.9241  | (est. ~0.91)| ~0.89 | 0.9982    | Lower F1 and recall; needs improvement     |
```

A point to note here is that the IoU estimates for U-Nets are based on loss, accuracy, and visual inspection as metric wasn't directly generated in the model pipeline.

Based on the given metrics, I can have a conclusion about the performance ranking with an in-depth analysis:

#### 1. **DeepLabV3+ (2nd Model)** – best performer

The 2nd model of DeepLabV3+ performs the best with the highest accuracy (99.72%) and F1 (0.9776) on validation set, as well as a strong mask overlap performance with IoU: 0.9578 on validation set. I configured the Training continued up to 30+ epochs and it shows consistent improvement, which suggests good learning dynamics. There is only minimal overfitting but a stable generalization to validation data. Therefore, in short, this version is the most ideal for MRI Tumor Segmentation where precision matters.

#### 2. **DeepLabV3+ (1st Model)** – strong, fast training

Th 1st model of DeepLabV3+ reaches 99.65% accuracy and 0.9724 F1 in just 10 epochs with a slightly lower IoU (0.9474 vs. 0.9578), but the metric is still robust. Another good point to mention is that the training loss converged quickly. This prompt convergence is beneficial when both training time and compute resources are limited. Therefore, it can be considered a fast deployment with high confidence.

#### 3. **DeepLabV3+ (3rd Model)** – still solid, slightly behind

The 3rd model of DeepLabV3+ sees very high accuracy and F1 at 99.57% and 0.9662, respectively. However, the training started with higher loss at 0.1751, suggesting a more difficult starting point. It also experiences lower convergence than previous two models due to slower learning rate and updated weight decay and decay rate, as well as a slight lower rate in IoU ~0.9342 on validation set. Therefore, it is more suitable if earlier models that I developed are not available or dataset noise increases.

#### 4. **Custom U-Net** – lightweight and impressive

The Custom U-Net delivers 99.58% accuracy and 0.9667 F1, which are comparable to DeepLabV3+ (3rd model). But, it is faster and has more interpretable architecture. It also experiences that the Val loss consistently drops, suggesting strong generalization. Therefore, it is a good option where compute or interpretability is a concern to have a resource-light model but still a trade-off between size and accuracy.

#### 5. **Xception-style U-Net** – underperformer

The Xception-stype U-Net has high metrics overall but lowest metrics among 5 Segmenters with Val F1 = 0.9241, Accuracy = 99.02% and estimated IoU at ~0.89.Although train accuracy is high, there is a larger gap between train/val F1, which suggests overfitting despite it might benefit from fewer layers, regularization and pretrained encoder weights. Therefore, it can be good as a prototype but is not good enough for production as-is.


--- 

# Conclusion

In my project, I developed a robust pipeline for Brain Tumor MRI Detection and Segmentation using multiple Deep Learning models. My experiments included Classification models based on ResNet50, EfficientNetV2L and ConvNextBase, and Segmentation models, including DeepLabV3+, Custom U-Net and Xception-stype U-Net. Based on my research and key findings, I found that ConvNeXtBase (2nd model) achieved the highest classification performance with near-perfect metrics with Accuracy ~100%, AUC = 1.00, IoU = 0.99, indicating excellent generalization. Besides, DeepLabV3+ (2nd model) provided the best Segmentation Accuracy (Val IoU = 0.9578), with stable validation performance. The robust Custom U-Net models showed lightweight efficiency, while Xception-style U-Net had lower Recall and requires architectural refinement for more perfect precision. I can prove my hypothesis that all of my models benefited from MUSICA image enhancement and data augmentation, improving both training stability and validation accuracy. My project demonstrates that transformer-inspired architectures like ConvNeXt and Atrous Spatial Pyramid Pooling (ASPP) in DeepLabV3+ are my state-of-the-art choices for tumor detection tasks. Moreover, using IoU as an auxiliary metric for both classification and segmentation enables better monitoring of spatial accuracy beyond simple accuracy or loss and other metrics like F1, precision, recall, ROC, AUC.

---

# Ideas for Future Work

Here are some of my ideas for scaling my project horizontally and vertically. As the MRI dataset I had was small and could affect the model's training and prediction, I can try using larger dataset such as BraTS 2021/2023 for more efficient training as a trade-off with compute costs and model architectural complexity. Fine-tuning on a greater dataset for benchmark comparison is also a meaningful enhancement. Furthermore, as I already implemented Vision Transformers (ViTs) and SAM (Segment Anything Model) in my pipelines, I can incorporate them for zero-shot or general-purpose segmentation. The dataset I worked with and developed models upon was static 2D MRIs. It would be even more beneficial to apply 3D MRI volumes with models like 3D U-Net or V-Net for volumetric segmentation.

---

# References

#### Medical Imaging, CAD, and AI

> **Hemanth, D. J., Anitha, J., & Pandian, J. A.** (2021). Brain tumor detection using hybrid deep learning techniques in MRI images. Scientific Reports, 11(1), 1-13. https://doi.org/10.1038/s41598-021-90428-8

> **Hemanth, D. J., Anitha, J., & Pandian, J. A.** (2023). Brain tumor detection and classification using machine learning and deep learning algorithms: A systematic review. BioMed Research International, 2023, Article ID 10453020. https://doi.org/10.1155/2023/10453020

> **Pianykh, O.S.** (2024). *Computer-Aided Diagnosis: From Feature Detection to AI in Radiology*. Lecture slides, CSCI E-87: Big Data and Machine Learning in Healthcare Applications, Harvard Extension School.  
> ➤ Key topics: CAD pipeline (feature extraction → classifier → validation), edge & line detection for fractures, convolution in LoG filters, multiscale resolution, PyRadiomics, CNNs for medical imaging.

Use this reference to support:
- Why ML is used in radiological images
- KMeans + morphological segmentation logic
- Convolutional methods (LoG, Gaussian, etc.)
- Discussion of PyRadiomics as an alternative or extension

> “Pathologies manifest as deviations from normal patterns; by extracting numeric features like shape, density, and texture, we can quantify abnormality — the essence of CAD.” — *Pianykh (2024)*

---

#### Multiscale Image Enhancement & MUSICA

> **Pianykh, O.S.** (2024). *Image Enhancement Techniques in Medical Imaging: From Denoising to CNNs*. Lecture slides, CSCI E-87: Big Data and Machine Learning in Healthcare Applications, Harvard Extension School.  
> ➤ Key topics: noise vs edges, bilateral filtering, Gaussian pyramids, Laplacian pyramids, multiscale decomposition and synthesis, MUSICA-style amplification.

Use this reference to support:
- Why using multiscale decomposition
- The basis for MUSICA-style image contrast enhancement
- Limitations of averaging vs adaptive multiscale amplification

> “By decomposing an image into low- and high-frequency components, and rebalancing with detail amplification, MUSICA enhances the contrast and diagnostic utility of the image.” — *Pianykh (2024)*

---

#### **Image Segmentation & Morphological Techniques**
1. **Morphological Snakes (Active Contour Models)**  
   *scikit-image team.*  
   _Morphological Snakes: Active contours without edges._  
   https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_morphsnakes.html

2. **Watershed Segmentation**  
   *scikit-image team.*  
   _Image segmentation using the watershed algorithm._  
   https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_watershed.html

3. **Morphological Operations in Image Processing**  
   *scikit-image team.*  
   _Denoising, sharpening and edge detection using kernel convolution._  
   https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html

4. **DeepLabV3+ Semantic Segmentation**

  *Keras Team.*

  _Image segmentation with DeepLabV3+._

  https://keras.io/examples/vision/deeplabv3_plus/

5. **Fully Convolutional Network (FCN) for Semantic Segmentation**

  *Keras Team.*

  _Image segmentation using a fully convolutional network._

  https://keras.io/examples/vision/fully_convolutional_network/

6. **BASNet for Background Matting and Segmentation**

  *Keras Team.*

  _Background matting using BASNet._

  https://keras.io/examples/vision/basnet_segmentation/
---

#### **Deep Learning with Keras**

4. **Segment Anything with SAM & Keras**  
   *Keras Team, 2024.*  
   _Segment Anything: Integrate SAM models for high-quality image segmentation._  
   https://keras.io/examples/vision/sam/

5. **Keras Applications (Pretrained CNNs)**  
   *Keras Documentation.*  
   _Keras Applications - Pretrained models like ResNet, VGG, Inception._  
   https://keras.io/api/applications/

6. **U-Net for Oxford Pets Dataset (Semantic Segmentation)**  
   *Keras Team, 2023.*  
   _Image segmentation of Oxford Pets using U-Net._  
   https://keras.io/examples/vision/oxford_pets_image_segmentation/

---

#### **Medical Image Dataset**

7. **Brain MRI Dataset for Tumor Detection (Kaggle)**  
   *Navoneel Chakrabarty, Kaggle.*  
   _Brain MRI Images for Brain Tumor Detection (Yes/No classification)._  
   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

---
### **Core Methodologies and Architectures**

1. **DeepLabV3+**

   > Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*. Proceedings of the European Conference on Computer Vision (ECCV), 801–818.
   > [https://arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)

2. **U-Net**

   > Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), Lecture Notes in Computer Science, vol. 9351, pp. 234–241. Springer.
   > [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

3. **EfficientNetV2**

   > Tan, M., & Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training*. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139:10096–10106.
   > [https://arxiv.org/abs/2104.00298](https://arxiv.org/abs/2104.00298)

4. **ConvNeXt (ConvNet + Transformer design)**

   > Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986.
   > [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

5. **Segment Anything (SAM)**

   > Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Schmidt, T., et al. (2023). *Segment Anything*. arXiv preprint.
   > [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

---

### **Preprocessing and Enhancement**

6. **MUSICA (Multiscale Image Contrast Amplification)**

   > Pizer, S. M., Eberly, D., Ericksen, J., & Hines, D. (1998). *Multiscale Image Contrast Amplification (MUSICA)*. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing.
   > [IEEE Paper Link](https://ieeexplore.ieee.org/document/4516995)

---

### **Datasets and Benchmarks**

7. **BraTS Challenge Dataset**

   > Menze, B. H., Jakab, A., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*. IEEE Transactions on Medical Imaging, 34(10), 1993–2024.
   > BraTS 2023 Official: [https://www.med.upenn.edu/cbica/brats2023/data.html](https://www.med.upenn.edu/cbica/brats2023/data.html)

---

### **Additional Suggested Reading**

8. **3D U-Net for Volumetric Segmentation**

   > Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*. In MICCAI 2016, LNCS, vol. 9901, pp. 424–432. Springer.
   > [https://arxiv.org/abs/1606.06650](https://arxiv.org/abs/1606.06650)

9. **Vision Transformers (ViTs)**

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.
> [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)




# 🧠🧬 Brain Tumor Deep Learning Detection (CNN, VGG, ResNet, EfficientNet, ConvNeXt) and Segmentation (Segment Anything Model (SAM), UNet-like Architecture, K-Means)

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

In my Brain Tumor Detection project, I chose this dataset from Kaggle with 115 brain imges with tumor and 98 without (clearly showing imbalance):

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

Data augmentation can help to apply certain tasks like rotation, flipping, contrast adjustment and noise surpression to have more diverse data for model training.

Class rebalancing might offer an effective fight against bias with oversampling or undersampling. I have to think which one makes more sense for this project.

Transfer learning can reduce much of model development by using pretrained models that were trained on large-scale medical datasets to improve performance on my limited data.

Not relying on a single metric, accuracy, is a must as there are other metrics like precision, recall and their curves, AUC ROC, F1 to consider about.

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

# Deep Learning (to be continued)
---

## Detection (to be continued)


### ResNet50 (initial test with 10 epochs)

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ resnet50 (Functional)           │ (None, 7, 7, 2048)     │    23,587,712 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 2048)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │       524,544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           257 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 24,112,513 (91.98 MB)
 Trainable params: 524,801 (2.00 MB)
 Non-trainable params: 23,587,712 (89.98 MB)
/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 43s 4s/step - accuracy: 0.5880 - loss: 1.2054 - val_accuracy: 0.8039 - val_loss: 0.3968
Epoch 2/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 163ms/step - accuracy: 0.8014 - loss: 0.5054 - val_accuracy: 0.9020 - val_loss: 0.1936
Epoch 3/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 178ms/step - accuracy: 0.8974 - loss: 0.2919 - val_accuracy: 0.9020 - val_loss: 0.2452
Epoch 4/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 160ms/step - accuracy: 0.8760 - loss: 0.2825 - val_accuracy: 0.9216 - val_loss: 0.1546
Epoch 5/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 162ms/step - accuracy: 0.9222 - loss: 0.1915 - val_accuracy: 0.9216 - val_loss: 0.2370
Epoch 6/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 158ms/step - accuracy: 0.9411 - loss: 0.2077 - val_accuracy: 0.9216 - val_loss: 0.2031
Epoch 7/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 161ms/step - accuracy: 0.9291 - loss: 0.1663 - val_accuracy: 0.9216 - val_loss: 0.1832
Epoch 8/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 153ms/step - accuracy: 0.9403 - loss: 0.1443 - val_accuracy: 0.9412 - val_loss: 0.1530
Epoch 9/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 163ms/step - accuracy: 0.9863 - loss: 0.0751 - val_accuracy: 0.9412 - val_loss: 0.1658
Epoch 10/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 1s 182ms/step - accuracy: 0.9901 - loss: 0.0749 - val_accuracy: 0.9216 - val_loss: 0.1543
```

![download (55)](https://github.com/user-attachments/assets/7ede18c8-7046-42d3-9e12-e57530028064)

![download (56)](https://github.com/user-attachments/assets/cd21b2cc-aa72-4d80-b7e1-dfc1cc5568ac)

### ResNet50 (2nd test with 100 epochs)

The ResNet50 model achieves very low training loss and high accuracy quickly but fails to improve validation metrics after 15–20 epochs. Also, validation loss increases, indicating memorization over learning, probably because of high model capacity, no regularization like dropout, weight decay and too gressive learning rate as well as lack of data and poor data augmentation at this stage.

![Resnet with 100 epochs plot 1](https://github.com/user-attachments/assets/c78a1b07-2853-4e6e-b6a5-66324afa7f45)

![Resnet with 100 epochs plot 2](https://github.com/user-attachments/assets/5b74b586-cc09-419b-85a5-f2331cdc2011)

### EfficientNetV2L (initial test with 20 epochs)

```
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ efficientnetv2-l (Functional)   │ (None, 10, 10, 1280)   │   117,746,848 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_1      │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 256)            │       327,936 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │           257 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 118,075,041 (450.42 MB)
 Trainable params: 328,193 (1.25 MB)
 Non-trainable params: 117,746,848 (449.17 MB)

Epoch 1/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 209s 17s/step - accuracy: 0.4944 - loss: 0.7149 - val_accuracy: 0.6667 - val_loss: 0.6336
Epoch 2/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 869ms/step - accuracy: 0.6148 - loss: 0.6604 - val_accuracy: 0.6471 - val_loss: 0.5832
Epoch 3/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 950ms/step - accuracy: 0.7113 - loss: 0.5747 - val_accuracy: 0.6863 - val_loss: 0.5738
Epoch 4/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 849ms/step - accuracy: 0.7207 - loss: 0.5724 - val_accuracy: 0.7059 - val_loss: 0.5405
Epoch 5/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 877ms/step - accuracy: 0.7270 - loss: 0.5381 - val_accuracy: 0.7059 - val_loss: 0.5365
Epoch 6/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 861ms/step - accuracy: 0.7605 - loss: 0.5045 - val_accuracy: 0.6863 - val_loss: 0.5297
Epoch 7/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 867ms/step - accuracy: 0.7999 - loss: 0.4878 - val_accuracy: 0.6667 - val_loss: 0.5135
Epoch 8/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 948ms/step - accuracy: 0.7605 - loss: 0.4920 - val_accuracy: 0.7451 - val_loss: 0.4829
Epoch 9/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 876ms/step - accuracy: 0.8005 - loss: 0.4526 - val_accuracy: 0.7647 - val_loss: 0.4693
Epoch 10/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 866ms/step - accuracy: 0.7923 - loss: 0.4734 - val_accuracy: 0.7647 - val_loss: 0.4662
Epoch 11/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 878ms/step - accuracy: 0.8028 - loss: 0.4228 - val_accuracy: 0.7647 - val_loss: 0.4596
Epoch 12/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 871ms/step - accuracy: 0.8170 - loss: 0.4092 - val_accuracy: 0.7647 - val_loss: 0.4536
Epoch 13/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 947ms/step - accuracy: 0.7970 - loss: 0.4020 - val_accuracy: 0.7647 - val_loss: 0.4385
Epoch 14/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 848ms/step - accuracy: 0.8466 - loss: 0.4057 - val_accuracy: 0.8235 - val_loss: 0.4320
Epoch 15/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 780ms/step - accuracy: 0.8466 - loss: 0.4097 - val_accuracy: 0.8235 - val_loss: 0.4398
Epoch 16/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 918ms/step - accuracy: 0.7961 - loss: 0.4263 - val_accuracy: 0.8627 - val_loss: 0.3862
Epoch 17/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 883ms/step - accuracy: 0.8566 - loss: 0.3769 - val_accuracy: 0.8431 - val_loss: 0.3809
Epoch 18/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 856ms/step - accuracy: 0.8111 - loss: 0.3892 - val_accuracy: 0.8627 - val_loss: 0.3855
Epoch 19/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 827ms/step - accuracy: 0.8027 - loss: 0.3922 - val_accuracy: 0.8235 - val_loss: 0.3963
Epoch 20/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 839ms/step - accuracy: 0.8794 - loss: 0.3494 - val_accuracy: 0.8824 - val_loss: 0.3652
Epoch 1/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 475s 29s/step - accuracy: 0.6904 - loss: 0.6123 - val_accuracy: 0.8235 - val_loss: 0.4042
Epoch 2/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 784ms/step - accuracy: 0.7314 - loss: 0.5867 - val_accuracy: 0.8431 - val_loss: 0.4646
Epoch 3/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 752ms/step - accuracy: 0.8559 - loss: 0.4764 - val_accuracy: 0.8039 - val_loss: 0.4711
Epoch 4/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 770ms/step - accuracy: 0.8466 - loss: 0.4569 - val_accuracy: 0.8235 - val_loss: 0.4267
Epoch 5/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 769ms/step - accuracy: 0.8791 - loss: 0.4253 - val_accuracy: 0.8431 - val_loss: 0.4566
Epoch 6/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 6s 779ms/step - accuracy: 0.8790 - loss: 0.4065 - val_accuracy: 0.8627 - val_loss: 0.4277
```
![download (57)](https://github.com/user-attachments/assets/0ba13af8-ed10-4a1c-ad28-a9493592f8bd)

![download (58)](https://github.com/user-attachments/assets/c2de14dd-e025-40f0-991f-7dcc3946a85b)


![download (59)](https://github.com/user-attachments/assets/df67b90e-c89f-4cca-a534-41b6d4e962ce)

![download (73)](https://github.com/user-attachments/assets/aeaf7a70-53d1-474f-aaeb-1cfd1b4a66a2)


### ConvNeXt Base (initial test with 20 epochs)

```
Epoch 1/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 63s 5s/step - accuracy: 0.5348 - loss: 0.8034 - val_accuracy: 0.7647 - val_loss: 0.5315
Epoch 2/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step - accuracy: 0.6786 - loss: 0.5861 - val_accuracy: 0.7843 - val_loss: 0.4466
Epoch 3/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - accuracy: 0.7181 - loss: 0.5227 - val_accuracy: 0.8627 - val_loss: 0.3961
Epoch 4/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step - accuracy: 0.7784 - loss: 0.4501 - val_accuracy: 0.9020 - val_loss: 0.3522
Epoch 5/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step - accuracy: 0.8601 - loss: 0.3578 - val_accuracy: 0.9020 - val_loss: 0.3243
Epoch 6/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step - accuracy: 0.8495 - loss: 0.3430 - val_accuracy: 0.9608 - val_loss: 0.3029
Epoch 7/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - accuracy: 0.8509 - loss: 0.3380 - val_accuracy: 0.9608 - val_loss: 0.2873
Epoch 8/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - accuracy: 0.8426 - loss: 0.3335 - val_accuracy: 0.9412 - val_loss: 0.2744
Epoch 9/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - accuracy: 0.8925 - loss: 0.2758 - val_accuracy: 0.9412 - val_loss: 0.2624
Epoch 10/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - accuracy: 0.9163 - loss: 0.2596 - val_accuracy: 0.9412 - val_loss: 0.2565
Epoch 11/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step - accuracy: 0.8917 - loss: 0.2690 - val_accuracy: 0.9412 - val_loss: 0.2488
Epoch 12/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - accuracy: 0.9224 - loss: 0.2534 - val_accuracy: 0.9412 - val_loss: 0.2426
Epoch 13/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - accuracy: 0.9049 - loss: 0.2505 - val_accuracy: 0.9412 - val_loss: 0.2448
Epoch 14/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - accuracy: 0.9216 - loss: 0.2375 - val_accuracy: 0.9412 - val_loss: 0.2355
Epoch 15/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - accuracy: 0.9324 - loss: 0.2117 - val_accuracy: 0.9412 - val_loss: 0.2334
Epoch 16/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - accuracy: 0.9153 - loss: 0.2043 - val_accuracy: 0.9412 - val_loss: 0.2319
Epoch 17/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - accuracy: 0.9352 - loss: 0.1994 - val_accuracy: 0.9412 - val_loss: 0.2237
Epoch 18/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step - accuracy: 0.9247 - loss: 0.2037 - val_accuracy: 0.9412 - val_loss: 0.2205
Epoch 19/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - accuracy: 0.9246 - loss: 0.2055 - val_accuracy: 0.9412 - val_loss: 0.2168
Epoch 20/20
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - accuracy: 0.9257 - loss: 0.1944 - val_accuracy: 0.9412 - val_loss: 0.2141
Epoch 1/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 132s 7s/step - accuracy: 0.9478 - loss: 0.1509 - val_accuracy: 0.9412 - val_loss: 0.1528
Epoch 2/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 293ms/step - accuracy: 0.9612 - loss: 0.1299 - val_accuracy: 0.9804 - val_loss: 0.1163
Epoch 3/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 290ms/step - accuracy: 0.9814 - loss: 0.0645 - val_accuracy: 0.9608 - val_loss: 0.1122
Epoch 4/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 291ms/step - accuracy: 0.9892 - loss: 0.0441 - val_accuracy: 0.9608 - val_loss: 0.1021
Epoch 5/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 291ms/step - accuracy: 1.0000 - loss: 0.0157 - val_accuracy: 0.9608 - val_loss: 0.0945
Epoch 6/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 292ms/step - accuracy: 1.0000 - loss: 0.0081 - val_accuracy: 0.9608 - val_loss: 0.0864
Epoch 7/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 287ms/step - accuracy: 1.0000 - loss: 0.0084 - val_accuracy: 0.9608 - val_loss: 0.0934
Epoch 8/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 289ms/step - accuracy: 1.0000 - loss: 0.0041 - val_accuracy: 0.9412 - val_loss: 0.1016
Epoch 9/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 289ms/step - accuracy: 1.0000 - loss: 0.0037 - val_accuracy: 0.9412 - val_loss: 0.1033
Epoch 10/10
7/7 ━━━━━━━━━━━━━━━━━━━━ 2s 288ms/step - accuracy: 1.0000 - loss: 0.0028 - val_accuracy: 0.9412 - val_loss: 0.1015
```
![download (74)](https://github.com/user-attachments/assets/c6b7dc92-60da-4a57-a679-ce4fbeb59bc9)

### ConvNeXt Base (2nd test with 100 epochs)

![Convnext with 100 epochs](https://github.com/user-attachments/assets/0a89e2f9-7a3e-4b07-861f-e016febf92c8)

The ConvNeXtBase experiences that the Train Accuracy is lightly lower but stable (~98–99%) and the Validation Accuracy is more stable at high 92–94% range while the Train Loss shows gradual and smoother decrease, and the Validation Loss illustrates a smooth trend with lower overfitting symptoms. These behaviors indicate a better generalization and less variance, therefore, the risk of overfitting is moderate to low as the Convergence Stability is evidenced with more consistent learning. However, before optimizing the model, I will need to apply these strategies for all models such as:

- Data Augmentation simulates larger dataset and adds regularization.
  
- EarlyStopping	prevents unnecessary epochs once validation loss or accuracy stagnates.
  
- Learning Rate Scheduler	assists model fine-tune without overshooting minima, such as Exponential Decay.

- Freezing Base Layers avoids destroying pretrained features too early.
  
- Reducing Learning Rate helps especially in the second half of training.


## Segmentation (to be continued)

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

--- 

# References

## References


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




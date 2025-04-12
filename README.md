# 🧠🧬 Brain Tumor Deep Learning Detection (CNN, VGG, ResNet, EfficientNet, ConvNeXt) and Segmentation (Segment Anything Model (SAM), UNet-like Architecture, K-Means)

![Harvard_University_logo svg](https://github.com/user-attachments/assets/0ea18127-d8c2-46ec-9f3e-10f2dc01d4d7)

![Harvard-Extension-School](https://github.com/user-attachments/assets/7de8c00d-6d74-456f-9b18-abb3174e83d5)

## **Master of Liberal Arts, Data Science**

## CSCI E-25 Computer Vision in Python

## Timeline: January 6th - May 16th, 2025 (In Progress)

## Professor: Stephen Elston

## Author: Dai-Phuong Ngo (Liam)

## First Words

In my **Brain Tumor Detection** project, I chose this dataset from Kaggle with 115 brain imges with tumor and 98 without (clearly showing imbalance): 

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

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

https://keras.io/examples/vision/deeplabv3_plus/

https://keras.io/examples/vision/basnet_segmentation/

https://keras.io/examples/vision/fully_convolutional_network/

This dataset is manageable in size, and I’ll rely on Keras, scikit-image and pretrained models to optimize both performance and training time.

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
---

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




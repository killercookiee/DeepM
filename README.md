# DeepM
DeepM is the future




## Project title: Exploring Vision Transformers and Hybrid Architectures for Medical Image Segmentation
Exploring Vision Transformers and Hybrid Architectures for Medical Image Segmentation
by Mohammed Salah Hamza Al-Radhi - Monday, 9 September 2024, 4:08 PM
Number of replies: 0
Transformer networks revolutionized deep learning, especially in the field of natural language processing, and are gaining much attention in computer vision. They offer great advantages over convolutional networks (CNN), such as higher flexibility, less sensitivity to hyperparameters, and the ability to effectively capture both local and global features in the input data. However, in some cases CNNs still excel, and for some tasks such as segmentation even transformer-based architectures use convolutional layers. Task of the students: explore transformer networks for medical image segmentation. Investigate open-source implementations, find and test a CNN-based baseline solution, and train 1-2 networks with different architectures on a cardiac MRI segmentation dataset (other, even non medica datasets also allowed). Pure-transformer or hybrid transformer-cnn or fully convolutional architectures are ok as well, one must be transformer based. Compare the selected networks by accuracy, throughput, sensitivity to hyperparameter changes, ease of implementation and training, etc. 

 ## Related materials: 

 https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb 
 https://www.cardiacatlas.org/lv-segmentation-challenge/
https://github.com/lxtGH/Awesome-Segmentation-With-Transformer#medical-image-segmentation 
 https://github.com/HuCaoFighting/Swin-Unet
https://arxiv.org/abs/2105.05537 
https://openaccess.thecvf.com/content/ICCV2021/html/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.html




## Group name: DeepM
Members:
1. Amin Hassairi + FVPKDV
2. Landolsi Hiba Allah + A8UNMW
3. Nguyen Ba Phi + S3VYH3
4. Praneshraj Tiruppur Nagarajan Dhyaneswar + AOMTO9



## File functions:

### 1st Milestone

#### /ACDC-dataset-preprocessing
- Just run DeepM_main_dataprep.ipynb to see how the data was prepared

    - /medical-image-segmentation         # This is the cloned repository of the dataset
    -    /dataset                        # This is the original dataset
    -    /MIS-working-dataset            # This is the dataset which is prepared

    - DeepM_main_dataprep.ipynb         # This is file which was used to prepare the data
    - DeepM_main_baseline.ipynb         # This is the file which will be the baseline
    - DeepM_main_baseline.ipynb         # This is the file which will be our model


### 2st & 3rd Milestone 

#### /Fully_CNN_Milestone_3
- Segmentation implemented using Fully Convolutional Network (FCN), developed from scratch. The ipynb file comprises loading the preprocessed data, FCN model and training with performance evaluation using metrics like IoU, Dice coefficient
- Please check the documentation file to comprehend better

#### /Transfuse_baseline
- ...

#### /UNETR_baseline
- Just run DeepM_main_UNETR.ipynb to train the model from scratch and save the best performed model.
  
    - Baseline UNETR model trained from scratch using the Synapse images https://www.synapse.org/#!Synapse:syn3193805/wiki/217752

#### /UNETR_fine_tuning_model
- Just run DeepM_main_FineTune_UNETR_model.ipynb to fine tune the model.

    - Using the pretrained model from the /UNETR_baseline, and fine tuning it with the preprocessed ACDC dataset https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb


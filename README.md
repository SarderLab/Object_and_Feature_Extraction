# Object_and_Feature_Extraction
This repository contains codes for extracting image patches from Whole Slide Image (WSI) annotations generated in Aperio ImageScope XML format, segmenting nuclei, and extracting image features for Glomeruli and Renal Tubules
- **ObjectExtraction:** For extracting image patches and segmenting nuclei from annotated tissue units
- **FeatureExtraction:** For measuring digital image features for glomeruli and tubules
# Data
Whole slide DN biopsies and annotation data can be found at athena.rc.ufl.edu
# Requirements
## **ObjectExtraction:**
- Tensorflow 1.14.0
- Python 3.6.5
- OpenSlide 3.4.0
- joblib 0.11
- imgaug 0.4.0
- imageio 2.3.0
- opencv 3.4.0
- PIL 5.3.0

## **FeatureExtraction:**
- Matlab (2021 or later)

# Usage
## **ObjectExtraction:**
Pretrained model files for nuclear segmentation can be found at ***IDK***

Edit the "xml_to_objects.py" script to properly point to directory with model file
```
model_dir = '/path/to/directory/with/model/files/'
```
Then, edit the annot_ID to correspond to which annotation layer you'd like to extract
```
annot_ID=2
```
Annotation IDs are as follows:
- Interstitium: annot_ID=1
- Glomeruli: annot_ID=2
- Globally-sclerotic glomeruli: annot_ID=3
- Tubules: annot_ID=4
- Arteries/Arterioles: annot_ID=5

Finally, run the script while pointing to directory with WSIs and XML annotations, the WSI extension (likely .svs), folder to output image crops, and whether to chop data (likely true)
```
python3 xml_to_objects.py --wsi /path/to/wsis/ --output /path/where/to/output/ --ext '.svs' --chop_data True
```

## **FeatureExtraction:**
Open the "Renal_quantification_master.m" file in Matlab

Change the input directory line in the code to the output directory from object extraction:
```
case_dir = '/path/of/objExt/output/'
```
Specify location and contents of label file:
```
excel_file = '/path/to/label/file.xlsx'
case_name_col = 'A'
data_range={'B','D'}
```
Labels should be in a format with:
- A column with case names that exactly match the folder names generated in object extraction
- Columns with label data (outcome, etc.)

Comment/Uncomment the name of the structure you wish to extract features for. For example, for glomeruli:
```
classname = '/Glomeruli/'
%classname = '/Sclerotic_glomeruli/'
%classname = '/Abnormal_tubule/'
```

When running the script for the first time, be sure to keep load mode set to 0:
```
load_mode=0
```
You may set annot_mode to 1 if you want a csv format output of the feature data:
```
annot_mode=1
```

Once you have run initial feature extraction for both glomeruli and globally sclerotic glomeruli, you can set:
```
load_mode=1
classname = '/Glomeruli_combined/'
```

This will aggregate features for both classes for a comprehensive set of glomerular features.

# Acknowledgements
Please cite our work @



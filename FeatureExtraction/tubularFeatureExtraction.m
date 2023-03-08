function [features]=tubularFeatureExtraction(segmentation_dir,image_dir)
% directory of glomerular component maps
segmented_gloms=dir([segmentation_dir,'/*.png']);
'Feature extraction'
Total=length(segmented_gloms);
features=zeros(Total,302);
% Remove small things smaller than this
min_object_size=25;
% stats=[];
parfor q=1:Total
    features(q,:)=feature_extraction_inner_tubule(q,segmented_gloms,image_dir,min_object_size);


end
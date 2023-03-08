function glomerularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,out_dir,classname)
'Glomerular compartment segmentation'

for g=1:length(image_dir)
    
    % Read image, glomerular mask, and nuclear mask
    I=imread(fullfile(image_dir(g).folder,image_dir(g).name));

    uID=strsplit(image_dir(g).name,'.jpeg');

    boundary=imread(fullfile(boundary_dir(g).folder,[uID{1,1},'_mask.png']))>0;
    nucSeg=imread(fullfile(nuc_dir(g).folder,[uID{1,1},'_mask.png']))>0;
    
%     boundary = imresize(boundary,4)>0;
%     I = imresize(I,4);
%     labbed = rgb2lab(I);
%     labbed(:,:,1) = 1.25*labbed(:,:,1);
%     I = lab2rgb(labbed);
%     I = imsharpen(I,'Amount',4);
    
%     [a,~,~] = colour_deconvolution(uint8(I),'H PAS');
%     a = 1-im2double(a);
%     nucSeg = a>0.45;
%     nucSeg = split_nuclei_functional(nucSeg);
    I = color_norm(I);
    I = uint8(255*I);

    HSV=rgb2hsv(I);

    saturation=imadjust(im2double(HSV(:,:,2)),[],[],0.7);%0.7
    WhiteSpaces=saturation < 0.3;
    WhiteSpaces=bwareaopen(WhiteSpaces,20);
    WhiteSpaces=imfill(WhiteSpaces,'holes');
    WhiteSpaces=imclose(WhiteSpaces,strel('disk',1));
      mes=~WhiteSpaces;
%     mes=saturation>0.3;%imbinarize(saturation,graythresh(saturation));

    nucSeg(~boundary)=0;
    

%     WhiteSpaces=~mes;
    WhiteSpaces(~boundary)=0;
    WhiteSpaces(nucSeg)=0;

    mes(~boundary)=0;
    mes(nucSeg)=0;

    
    final_mask=cat(3,mes,WhiteSpaces,nucSeg);
    final_mask(~repmat(boundary,[1,1,3]))=0;
    
    %shrink again
%     final_mask = imresize(final_mask,0.25);
    
%     figure(1),imshow(im2double(final_mask))
%     figure(2),imshow(I),pause,continue
    imwrite(double(final_mask),[out_dir,'/',uID{1,1},'.png'])


end
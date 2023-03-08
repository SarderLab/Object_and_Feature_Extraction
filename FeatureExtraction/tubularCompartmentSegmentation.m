function tubularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,out_dir,classname)
'Tubular compartment segmentation'

parfor g=1:length(image_dir)
    I=imread(fullfile(image_dir(g).folder,image_dir(g).name));

    uID=strsplit(image_dir(g).name,'.jpeg');
    boundary=imread(fullfile(boundary_dir(g).folder,[uID{1,1},'_mask.png']))>0;
    nucSeg=imread(fullfile(nuc_dir(g).folder,[uID{1,1},'_mask.png']))>0;
    
%     [a,~,~] = colour_deconvolution(uint8(I),'H PAS');
%     a = 1-im2double(a);
%     nucSeg = a>graythresh(a);%0.45;
%     nucSeg = split_nuclei_functional(nucSeg);
    
%     nucSeg = nucSeg+nucSeg2 > 0;
    I = color_norm(I);
    I=uint8(255*I);

    LAB=im2double(rgb2lab(I));

    lightness=(LAB(:,:,1));

%     WhiteSpaces=lightness>80;
    
    %FFPE
%     lightness = lightness / 71.5;
%     WhiteSpaces = lightness > 0.61;
    %LNR01
    lightness = lightness/100;
    WhiteSpaces = lightness > 0.88;
    WhiteSpaces=bwareaopen(WhiteSpaces,20);
    WhiteSpaces=imfill(WhiteSpaces,'holes');
    WhiteSpaces=imclose(WhiteSpaces,strel('disk',1));

    mes=~WhiteSpaces;
    nucSeg(~boundary)=0;
    
    WhiteSpaces(~boundary)=0;
    WhiteSpaces(nucSeg)=0;

    mes(~boundary)=0;
    mes(nucSeg)=0;

    final_mask=cat(3,mes,WhiteSpaces,nucSeg);
    final_mask(~repmat(boundary,[1,1,3]))=0;
%     figure,subplot(121),imshow(I)
%     subplot(122),imshow(im2double(final_mask)),pause
    imwrite(double(final_mask),[out_dir,'/',uID{1,1},'.png'])
end
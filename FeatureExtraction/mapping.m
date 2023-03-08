filename = '/blue/pinaki.sarder/nlucarelli/ObjectExtraction/DN/31_S_4/Abnormal_tubule/Images/31_S_4_1305.jpeg';
compname = '/blue/pinaki.sarder/nlucarelli/ObjectExtraction/DN/31_S_4/Abnormal_tubule/CompartmentSegmentations/31_S_4_1305.png';

im = imread(filename);
im = color_norm(im);
im = uint8(255*im);
comp = imread(compname);

mes = comp(:,:,1)>0;
boundary_mask = (comp(:,:,1)+comp(:,:,2)+comp(:,:,3))>0;



%Specific here
[~,sat,~]=colour_deconvolution(im,'H PAS');
sat=1-im2double(sat);
sat=imadjust(sat,[],[],3);
boundary_w_mem=imdilate(boundary_mask,strel('disk',10));
mems=imbinarize(sat,adaptthresh(sat,0.3));
blim=boundary_w_mem;
indel=imerode(blim,strel('disk',10));
% figure,imshow(mems)
blim(indel)=0;
tbm=imreconstruct(blim&mems,mems);
tbm(~boundary_w_mem)=0;
tbm=bwareaopen(tbm,50);
tbm=imclose(tbm,strel('disk',1));

labbed = rgb2lab(im);
labbed(:,:,1) = labbed(:,:,1)*0.5;
back = lab2rgb(labbed);

im_masked = im2double(im).*tbm;
back = back - back.*tbm;


im_masked = tbm;

% dt = bwdist(~mes);
% imagesc(dt);
% axis off;axis image;
% ax = gca;
% exportgraphics(ax,'/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/sample.png')
% dt = imread('/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/sample.png');
% dt = imresize(dt,size(mes));
% dt = im2double(dt).*mes;
% im_masked = dt;

back = back + im_masked;

imshow(back)
imwrite(back,'/blue/pinaki.sarder/nlucarelli/FeatureExtraction/DN/tub_feature_high.png');
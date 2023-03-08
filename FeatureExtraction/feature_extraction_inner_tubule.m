function v=feature_extraction_inner_tubule(q,segmented_gloms,image_dir,min_object_size)
% 
% disp(q)

v=zeros(1,302);

nucpixradius=2;

composite=imread(fullfile(segmented_gloms(q).folder,segmented_gloms(q).name))>0;
I=imread(fullfile(image_dir(q).folder,image_dir(q).name));
I=color_norm(I);
I=uint8(255*I);


mes_mask=composite(:,:,1);
white_mask=composite(:,:,2);

nuc_mask=composite(:,:,3);
nuc_mask=bwareaopen(nuc_mask,2);
boundary_mask=mes_mask|white_mask|nuc_mask;
boundary_mask=bwpropfilt(boundary_mask,'Area',1);

mes_mask=~bwareaopen(~mes_mask,min_object_size);
mes_mask=bwareaopen(mes_mask,min_object_size);

white_mask=bwareaopen(white_mask,min_object_size);
white_mask=imfill(white_mask,'holes');

boundary_w_mem=imdilate(boundary_mask,strel('disk',10));

gdist=bwdist(~boundary_mask);


gdist=(-1*(gdist))+max(gdist(:));
gdist(~boundary_mask)=0;


gArea=sum(boundary_mask(:));

gOutline=bwperim(boundary_mask);

[r,c]=find(boundary_mask);
rMean=round(mean(r));
cMean=round(mean(c));



gdist2=zeros(size(boundary_mask));

gdist2(~boundary_mask)=1;
gdist2=bwdist(gdist2);
gdist2=(gdist2-max(max(gdist2)))*-1;
gdist2(~boundary_mask)=0;


nucdist=gdist;
nucdist(~nuc_mask)=0;
nuc_areas=regionprops(nuc_mask,'SubarrayIdx');

uobs=bwlabel(nuc_mask);
statsl=[];
for i=1:numel(nuc_areas)
    loc=nuc_areas(i).SubarrayIdx;
    smallmask=uobs(loc{:});

    dtvals=nucdist(loc{:}).*double(smallmask==i);
    if sum(dtvals(:))==0
        ovals=[0,0,0];
    else
        ovals=[mean(dtvals(:)),min(dtvals(dtvals>0)),max(dtvals(:))];
    end

    statsl=[statsl;ovals];

    
end




nuc_dist_bound=double(gdist2.*double(nuc_mask));
lum_dist_bound=double(gdist2.*double(white_mask));
mes_dist_bound=double(gdist2.*double(mes_mask));

lv=quantile(nonzeros(lum_dist_bound(:)),[.1:.1:1]);
nv=quantile(nonzeros(nuc_dist_bound(:)),[.1:.1:1]);
mv=quantile(nonzeros(mes_dist_bound(:)),[.1:.1:1]);

mes=mes_mask;
mes=bwareaopen(mes,min_object_size);

[~,sat,~]=colour_deconvolution(I,'H PAS');
sat=1-im2double(sat);
sat=imadjust(sat,[],[],3);


mems=imbinarize(sat,adaptthresh(sat,0.3));
blim=boundary_w_mem;
indel=imerode(blim,strel('disk',10));
% figure,imshow(mems)
blim(indel)=0;
tbm=imreconstruct(blim&mems,mems);
tbm(~boundary_w_mem)=0;
tbm=bwareaopen(tbm,50);
tbm=imclose(tbm,strel('disk',1));
% figure,imshow(tbm)

fibers=fibermetric(sat,2:4:20);
inmem=fibers>0.6;
inmem(tbm)=0;
inmem(~boundary_w_mem)=0;
inmem(nuc_mask)=0;
%inmem=bwareaopen(inmem,50);
% figure,imshow(inmem),pause
% [y,x]=find(nuc_mask);
% [theta,rhoN]=cart2pol(x-rMean,y-cMean);
% N_n=histcounts(rhoN,[0:100:1000,2500]);
% T_n=histcounts(theta,[-pi:(2*pi/20):pi]);
% 
% 
% [y,x]=find(inmem);
% [~,rho]=cart2pol(x-rMean,y-cMean);
% N_l=histcounts(rho,[0:100:1000,2500]);

inmemdist=gdist;
inmemdist(~inmem)=0;
inmem_areas=regionprops(inmem,'SubarrayIdx');

uobs=bwlabel(inmem);
statst=[];
for i=1:numel(inmem_areas)
    loc=inmem_areas(i).SubarrayIdx;
    smallmask=uobs(loc{:});

    dtvals=inmemdist(loc{:}).*double(smallmask==i);
    if sum(dtvals(:))==0
        ovals=[0,0,0];
    else
        ovals=[mean(dtvals(:)),min(dtvals(dtvals>0)),max(dtvals(:))];
    end


    statst=[statst;ovals];

    
end

% [y,x]=find(tbm);
% [~,rho]=cart2pol(x-rMean,y-cMean);
% N_m=histcounts(rho,[0:100:1000,2500]);
tbmdist=gdist;
tbmdist(~tbm)=0;
tbm_areas=regionprops(tbm,'SubarrayIdx');

% figure,imshow(nuc_mask)
uobs=bwlabel(tbm);
statstbm=[];
for i=1:numel(tbm_areas)
    loc=tbm_areas(i).SubarrayIdx;
    smallmask=uobs(loc{:});

    dtvals=tbmdist(loc{:}).*double(smallmask==i);
    if sum(dtvals(:))==0
        ovals=[0,0,0];
    else
        ovals=[mean(dtvals(:)),min(dtvals(dtvals>0)),max(dtvals(:))];
    end
    statstbm=[statstbm;ovals];

    
end


mdt=bwdist(~mes);
%%Flag
% figure(1),subplot(131),imshow(I),subplot(132),imshow(white_mask),subplot(133),imshow(imfill(white_mask,'holes')),pause,return
ldt=bwdist(~white_mask);
% edges=[1:1:60,2000];
% N2=histcounts(ldt(ldt(:)>0),edges);

wmask=imfill(white_mask,'holes');
% lumdt=bwdist(~wmask);
% white_areas=regionprops(wmask,'SubarrayIdx');
% uobs=bwlabel(wmask);
% statsl=[];
% for i=1:numel(white_areas)
%     loc=white_areas(i).SubarrayIdx;
%     smallmask=uobs(loc{:});
%     dtvals=lumdt(loc{:}).*double(smallmask==i);
%     statsl=[statsl,max(dtvals(:))];
% end
% hist1=hist(statsl,[0:2:80,90:20:170]);
% v=statsl;
% return

ndt=bwdist(~nuc_mask);



grayIm=rgb2gray(I);
grayIm(~mes_mask)=NaN;

[ratiosM,s1,mes_num]=getCompRatios(composite,grayIm,min_object_size);


composite=cat(3,white_mask,mes_mask,nuc_mask);
composite(~repmat(boundary_mask,[1,1,3]))=0;


grayIm=rgb2gray(I);
grayIm(~white_mask)=NaN;

[ratiosL,s2,lum_num]=getCompRatios(composite,grayIm,min_object_size);

composite=cat(3,nuc_mask,white_mask,mes_mask);
composite(~repmat(boundary_mask,[1,1,3]))=0;
grayIm=rgb2gray(I);
grayIm(~nuc_mask)=NaN;


[ratiosN,s3,nuc_num]=getNucRatios(composite,nucpixradius,grayIm);

distsN=getCompDists(nuc_mask,gOutline,[rMean,cMean]);
% distsM=getCompDists(mes_mask,gOutline,[rMean,cMean]);
% distsL=getCompDists(white_mask,gOutline,[rMean,cMean]);
diststbm=getCompDists(tbm,gOutline,[rMean,cMean]);
distsinmem=getCompDists(inmem,gOutline,[rMean,cMean]);

v(1,1:3)=mean(ratiosL(:,1:3));
v(1,4)=sum(ratiosL(:,4));
v(1,5)=mean(ratiosL(:,4));
v(1,6)=median(ratiosL(:,4));


v(1,7:10)=[s1(1,1).Contrast,s1(1,2).Correlation,s1(1,3).Energy,s1(1,4).Homogeneity];

v(1,11:13)=mean(ratiosM(:,1:3));
v(1,12)=mean(ratiosM(:,2));
v(1,13)=mean(ratiosM(:,3));
v(1,14)=sum(ratiosM(:,4));
v(1,15)=mean(ratiosM(:,4));
v(1,16)=median(ratiosM(:,4));


v(1,17:20)=[s2(1,1).Contrast,s2(1,2).Correlation,s2(1,3).Energy,s2(1,4).Homogeneity];


v(1,21)=mean(ratiosN(:,1));
v(1,22)=mean(ratiosN(:,2));
v(1,23)=mean(ratiosN(:,3));

v(1,24)=sum(ratiosN(:,4));
v(1,25)=mean(ratiosN(:,4));
v(1,26)=mode(ratiosN(:,4));

v(1,27:30)=[s3(1,1).Contrast,s3(1,2).Correlation,s3(1,3).Energy,s3(1,4).Homogeneity];

v(1,31:37)=mean(distsN);

v(1,38)=gArea;

v(1,39)=mes_num;
v(1,40)=lum_num;
v(1,41)=nuc_num;

[d1,d2,~]=size(I);
thinfeats=get_thinness(tbm);
dt=bwdist(~tbm);
grayIm=rgb2gray(I);
grayIm(~tbm)=NaN;
stats=graycoprops(graycomatrix(grayIm));

tbm_int=im2double(I);
tbm_int(~repmat(tbm,[1,1,3]))=NaN;
tbm_int=reshape(tbm_int,[d1*d2,3]);


if ~isempty(thinfeats)
    
    v(1,42)=mean(thinfeats);
    v(1,43)=max(thinfeats);
    v(1,44)=min(thinfeats);
    v(1,45)=mean(mean(dt));
    v(1,46)=max(max(dt));
    v(1,47)=stats.Energy;
    v(1,48)=stats.Correlation;
    v(1,49)=stats.Contrast;
    v(1,50)=stats.Homogeneity;
    
    stats=regionprops(tbm,'solidity');
    v(1,51)=mean(tbm_int(:,1),'omitnan');
    v(1,52)=mean(tbm_int(:,2),'omitnan');
    v(1,53)=mean(tbm_int(:,3),'omitnan');
    v(1,54)=std(tbm_int(:,1),[],'omitnan');
    v(1,55)=std(tbm_int(:,2),[],'omitnan');
    v(1,56)=std(tbm_int(:,3),[],'omitnan');
    
    v(1,57)=sum(sum(tbm));
    v(1,58)=mean([stats.Solidity]);
    v(1,59:65)=mean(diststbm);

end

thinfeats=get_thinness(inmem);
dt=bwdist(~inmem);
grayIm=rgb2gray(I);
grayIm(~inmem)=NaN;
stats=graycoprops(graycomatrix(grayIm));
inmem_int=im2double(I);
inmem_int(~repmat(inmem,[1,1,3]))=NaN;
inmem_int=reshape(inmem_int,[d1*d2,3]);

if ~isempty(thinfeats)
    v(1,66)=mean(thinfeats);
    v(1,67)=max(thinfeats);
    v(1,68)=min(thinfeats);
    v(1,69)=mean(mean(dt));
    v(1,70)=mean(mean(dt));
    v(1,71)=stats.Energy;
    v(1,72)=stats.Correlation;
    v(1,73)=stats.Contrast;
    v(1,74)=stats.Homogeneity;
    
     stats=regionprops(inmem,'solidity');
    v(1,75)=mean(inmem_int(:,1),'omitnan');
    v(1,76)=mean(inmem_int(:,2),'omitnan');
    v(1,77)=mean(inmem_int(:,3),'omitnan');
    v(1,78)=std(inmem_int(:,1),[],'omitnan');
    v(1,79)=std(inmem_int(:,2),[],'omitnan');
    v(1,80)=std(inmem_int(:,3),[],'omitnan');
    v(1,81)=std(inmem_int(:,3),[],'omitnan');
    
    v(1,82)=sum(sum(inmem));
    v(1,83)=mean([stats.Solidity]);
    v(1,84:90)=mean(distsinmem);
end

% imwrite(double(inmem),[fullfile(image_dir(q).folder,image_dir(q).name),'_feat.png'])
% v(1,91:91+45)=hist1;
% return
tubule_morphology=regionprops(boundary_mask,'Area','Eccentricity','MajorAxisLength','MinorAxisLength','Perimeter','Solidity');

v(1,137)=(4*pi*tubule_morphology.Area)/(tubule_morphology.Perimeter.^2);
v(1,138)=tubule_morphology.Eccentricity;
v(1,139)=sqrt(4*tubule_morphology.Area*pi);
v(1,140)=tubule_morphology.MajorAxisLength;
v(1,141)=tubule_morphology.MinorAxisLength;
v(1,142)=tubule_morphology.Perimeter;
%fiber length
v(1,143)=real((tubule_morphology.Perimeter-sqrt(tubule_morphology.Perimeter.^2-(16*tubule_morphology.Area)))/4);

%fiber width
v(1,144)=tubule_morphology.Area/real((tubule_morphology.Perimeter-sqrt(tubule_morphology.Perimeter.^2-(16*tubule_morphology.Area)))/4);
%curl
v(1,145)=tubule_morphology.MajorAxisLength/real((tubule_morphology.Perimeter-sqrt(tubule_morphology.Perimeter.^2-(16*tubule_morphology.Area)))/4);
v(1,146)=tubule_morphology.Solidity;


mes_int=im2double(I);
mes_int(~repmat(mes,[1,1,3]))=NaN;
mes_int=reshape(mes_int,[d1*d2,3]);

lum_int=im2double(I);
lum_int(~repmat(white_mask,[1,1,3]))=NaN;
lum_int=reshape(lum_int,[d1*d2,3]);

nuc_int=im2double(I);
nuc_int(~repmat(nuc_mask,[1,1,3]))=NaN;
nuc_int=reshape(nuc_int,[d1*d2,3]);

v(1,147)=mean(mes_int(:,1),'omitnan');
v(1,148)=mean(mes_int(:,2),'omitnan');
v(1,149)=mean(mes_int(:,3),'omitnan');
v(1,150)=std(mes_int(:,1),[],'omitnan');
v(1,151)=std(mes_int(:,2),[],'omitnan');
v(1,152)=std(mes_int(:,3),[],'omitnan');

v(1,153)=mean(lum_int(:,1),'omitnan');
v(1,154)=mean(lum_int(:,2),'omitnan');
v(1,155)=mean(lum_int(:,3),'omitnan');
v(1,156)=std(lum_int(:,1),[],'omitnan');
v(1,157)=std(lum_int(:,2),[],'omitnan');
v(1,158)=std(lum_int(:,3),[],'omitnan');

v(1,159)=mean(nuc_int(:,1),'omitnan');
v(1,160)=mean(nuc_int(:,2),'omitnan');
v(1,161)=mean(nuc_int(:,3),'omitnan');
v(1,162)=std(nuc_int(:,1),[],'omitnan');
v(1,163)=std(nuc_int(:,2),[],'omitnan');
v(1,164)=std(nuc_int(:,3),[],'omitnan');

if sum(lum_dist_bound(:))>0
    v(1,165)=min(lum_dist_bound(lum_dist_bound(:)>0));
    v(1,166)=max(lum_dist_bound(:));
    v(1,167)=mean(lum_dist_bound(lum_dist_bound(:)>0));
    v(1,168)=median(lum_dist_bound(lum_dist_bound(:)>0));
    v(1,169)=std(lum_dist_bound(lum_dist_bound(:)>0));
end

if sum(mes_dist_bound(:))>0
    v(1,170)=min(mes_dist_bound(mes_dist_bound(:)>0));
    v(1,171)=max(mes_dist_bound(:));
    v(1,172)=mean(mes_dist_bound(mes_dist_bound(:)>0));
    v(1,173)=median(mes_dist_bound(mes_dist_bound(:)>0));
    v(1,174)=std(mes_dist_bound(mes_dist_bound(:)>0));
end
% v(1,165:175)=N_n;
% 
% v(1,176:186)=N_l;
% 
% v(1,187:197)=N_m;
% 
% v(1,198:217)=T_n;

v(1,218:227)=nv;
v(1,228:237)=mv;
v(1,238:247)=lv;

v(1,248)=max(mdt(:));
v(1,249)=max(ldt(:));
v(1,250)=max(ndt(:));
v(1,251)=max(max(gdist));
% v(1,252)=mean(T_n);
% v(1,253)=max(T_n);
% v(1,254)=min(T_n);
% if ~isempty(rhoN)
%     v(1,255)=mean(rhoN);
%     v(1,256)=max(rhoN);
%     v(1,257)=min(rhoN);
% end
if numel(nuc_areas)>0
v(1,258)=max(statsl(:,1));
v(1,259)=max(statsl(:,2));
v(1,260)=max(statsl(:,3));


v(1,261)=mean(statsl(:,1));
v(1,262)=mean(statsl(:,2));
v(1,263)=mean(statsl(:,3));
v(1,264)=min(statsl(:,1));
v(1,265)=min(statsl(:,2));
v(1,266)=min(statsl(:,3));
v(1,267)=var(statsl(:,1));
v(1,268)=var(statsl(:,2));
v(1,269)=var(statsl(:,3));
v(1,270)=median(statsl(:,1));
v(1,271)=median(statsl(:,2));
v(1,272)=median(statsl(:,3));
end

if numel(inmem_areas)>0
v(1,273)=max(statst(:,1));
v(1,274)=max(statst(:,2));
v(1,275)=max(statst(:,3));
v(1,276)=mean(statst(:,1));
v(1,277)=mean(statst(:,2));
v(1,278)=mean(statst(:,3));
v(1,279)=min(statst(:,1));
v(1,280)=min(statst(:,2));
v(1,281)=min(statst(:,3));
v(1,282)=var(statst(:,1));
v(1,283)=var(statst(:,2));
v(1,284)=var(statst(:,3));
v(1,285)=median(statst(:,1));
v(1,286)=median(statst(:,2));
v(1,287)=median(statst(:,3));
end
if numel(tbm_areas)>0
v(1,288)=max(statstbm(:,1));
v(1,289)=max(statstbm(:,2));
v(1,290)=max(statstbm(:,3));
v(1,291)=mean(statstbm(:,1));
v(1,292)=mean(statstbm(:,2));
v(1,293)=mean(statstbm(:,3));
v(1,294)=min(statstbm(:,1));
v(1,295)=min(statstbm(:,2));
v(1,296)=min(statstbm(:,3));
v(1,297)=var(statstbm(:,1));
v(1,298)=var(statstbm(:,2));
v(1,299)=var(statstbm(:,3));
v(1,300)=median(statstbm(:,1));
v(1,301)=median(statstbm(:,2));
v(1,302)=median(statstbm(:,3));
end
end
function [normalized] = color_norm(source_glom)

%RGB = 201.973,180.509,209.384
%STD = 36.115,44.521,31.570

%DN-SK
stats = zeros(2,3);
stats(1,1) = 75.86;%
stats(1,2) = 13.238;%
stats(1,3) = -11.870;%
stats(2,1) = 15.174;%
stats(2,2) = 8.348;%
stats(2,3) = 7.593;%

%LNR01
% stats = zeros(2,3);
% stats(1,1) = 65.00;%
% stats(1,2) = 18.8;%
% stats(1,3) = -14.9;%
% stats(2,1) = 15.25;%
% stats(2,2) = 9.522;%
% stats(2,3) = 7.73;

m_l = stats(1,1);
m_a = stats(1,2);
m_b = stats(1,3);

s_l = stats(2,1);
s_a = stats(2,2);
s_b = stats(2,3);

source1 = rgb2lab(im2double(source_glom));

sourceL = source1(:,:,1);
sourceA = source1(:,:,2);
sourceB = source1(:,:,3);

[x,y,z] = size(source1);

sourceL=reshape(sourceL,1,x*y);
sourceL=double(sourceL);
sourceA=reshape(sourceA,1,x*y);
sourceA=double(sourceA);
sourceB=reshape(sourceB,1,x*y);
sourceB=double(sourceB);
std1=std(sourceL,1);          %Finding out standard deviation of individual 
std2=std(sourceA,1);          %channels of source image
std3=std(sourceB,1);
m1=mean(mean(source1(:,:,1))); %Finding out mean of individual channels 
m2=mean(mean(source1(:,:,2))); %of source image
m3=mean(mean(source1(:,:,3)));

normalized = zeros(x,y,3);

for i=1:x
    for j=1:y
        
        normalized(i,j,1) = ((source1(i,j,1)-m1)*(s_l/std1))+m_l;
        normalized(i,j,2) = ((source1(i,j,2)-m2)*(s_a/std2))+m_a;
        normalized(i,j,3) = ((source1(i,j,3)-m3)*(s_b/std3))+m_b;
        
    end
end

normalized = lab2rgb(normalized);


end
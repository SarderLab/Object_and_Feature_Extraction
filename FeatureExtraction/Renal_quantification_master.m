close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Developed by Brandon Ginley while researching as a PhD candidate in the 
% lab of Pinaki Sarder at the SUNY Jacobs School of Medicine and Biomedical
% Sciences.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Patient folders
% case_dir=uigetdir();
case_dir = '/path/to/extracted/cases/';

% Where to save data exported for seurat
seurat_data_output='/where/to/save/feature/outputs/';

% Create label data for seurat analysis
annot_mode=1; % Whether or not to create
excel_file='/path/to/label/file.xlsx'; % Label file 
case_name_col='A';% Column with case names
data_range={'B','D'};% Columns with label data % DN

proteomics_file = '/path/to/proteomics/file.xlsx';%Put case names in column A
proteomics_data_range={'B','AT'};


% If already run once, turn to 1 to skip feature re-creation
% And write features to txt file
load_mode=0;

biopsy_cases=dir(case_dir);
biopsy_cases(1:2)=[];
dirFlags=[biopsy_cases.isdir];
biopsy_cases=biopsy_cases(dirFlags);

%Select object to extract
% classname='/Abnormal_tubule/';
classname='/Glomeruli/';
% classname='/Sclerotic_glomeruli/';

%%Only for load mode
% classname='/Glomeruli_combined/';
c=strsplit(classname,'/');
c=c{2};

if ~exist(seurat_data_output,'dir')
   mkdir(seurat_data_output) 
end
txt=fopen([fullfile(seurat_data_output,c),'_features.txt'],'w');
ordertxt=fopen([fullfile(seurat_data_output,c),'_order.txt'],'w');

features_full=[];
items_per_case = [];
for b_c=1:numel(biopsy_cases)
    case_ID=biopsy_cases(b_c).name;
    display(['Working on case ' case_ID])
    
    image_dir=dir(fullfile(case_dir,case_ID,classname,'/Images/*.jpeg'));
    boundary_dir=dir(fullfile(case_dir,case_ID,classname,'/Boundary_segmentations/*.png'));
    nuc_dir=dir(fullfile(case_dir,case_ID,classname,'/Nuclear_segmentations/prediction/*.png'));
    if load_mode==0
        processDeepLabSegmentations(nuc_dir)
    end
    %Directory for saving comparmtent segmentation data
    segment_out_dir=fullfile(case_dir,case_ID,classname,'/CompartmentSegmentations');
    if ~exist(segment_out_dir)
        mkdir(segment_out_dir)
    end
    
    %Passed in directory for saving image visualizations
    
    if load_mode==0
        
        if strcmp(classname,'/Abnormal_tubule/')
            tubularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,segment_out_dir,classname)
            features=tubularFeatureExtraction(segment_out_dir,image_dir);
%               features = zeros(1,207);

        elseif strcmp(classname,'/Glomeruli/')||strcmp(classname,'/Sclerotic_glomeruli/')
            glomerularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,segment_out_dir,classname)
            features=glomerularFeatureExtraction(segment_out_dir,image_dir);
%               features = zeros(1,315);
        end
    
    end
    % Directory to save feature data
    sdir=fullfile(case_dir,case_ID,classname,'Features');
    if ~exist(sdir)
        mkdir(sdir)
    end
    
    
    if load_mode==0
        sdirm=fullfile(case_dir,case_ID,classname,'Features',[case_ID,'.mat']);
        save(sdirm,'features')
    else
        if strcmp(classname,'/Glomeruli/')||strcmp(classname,'/Sclerotic_glomeruli/')||strcmp(classname,'/Abnormal_tubule/')||strcmp(classname,'/Widened_Interstitium/')||strcmp(classname,'/Vessel/')
            sdirm=fullfile(case_dir,case_ID,classname,'Features',case_ID);
            load(sdirm,'features')
            features(isnan(features))=0;
            [n_tot,~] = size(features);
            if strcmp(classname,'/Abnormal_tubule/')
                load missing_index
                features(:,missing_index)=[];
                [n_tot,~] = size(features);
            end

            writetextfeatures(txt,ordertxt,b_c,image_dir,features)
        else
            sdirm=fullfile(case_dir,case_ID,'/Glomeruli/','Features',case_ID);
            load(sdirm,'features')
            features(isnan(features))=0;
            features(isinf(features))=10000;
            gf=features;
            [n_non,~] = size(gf); 
            sdirm=fullfile(case_dir,case_ID,'/Sclerotic_glomeruli/','Features',case_ID);
            load(sdirm,'features')
            features(isnan(features))=0;
            features(isinf(features))=10000;
            [n_scler,~] = size(features);
            n_tot = n_non + n_scler;
            features=cat(1,gf,features);
            
            temp_dir1=dir(fullfile(case_dir,case_ID,'/Glomeruli','/Images/*.jpeg'));
            temp_dir2=dir(fullfile(case_dir,case_ID,'/Sclerotic_glomeruli','/Images/*.jpeg'));

            temp_dir = [temp_dir1;temp_dir2];
            
            writetextfeatures(txt,ordertxt,b_c,temp_dir,features)
        end
        features_full=[features_full;features];
        items_per_case = [items_per_case;n_tot];
        
    end
end
fclose('all')
%Save features
save(fullfile(case_dir,[c,'_full.mat']),'features_full')


if annot_mode==1
    % Take care of rare cases where zero detected nuclei or tubules creates
    % undefined feature measurements
    features_full(isnan(features_full))=0;
    features_full(isinf(features_full))=10000;
    labeled_matrix=get_label_data(excel_file,case_dir,classname,case_name_col,data_range,items_per_case);
    confounder
%     proteomics_matrix = get_label_data(proteomics_file,case_dir,classname,case_name_col,proteomics_data_range,items_per_case);
       
    
    features_write=cat(2,[1:1:size(features_full,1)]',features_full)';
    
    
    csvwrite([fullfile(seurat_data_output,c),'_features.csv'],features_write);
    csvwrite([fullfile(seurat_data_output,c),'_labels.csv'],labeled_matrix);
%     csvwrite([fullfile(seurat_data_output,c),'_sig_proteomics.csv'],proteomics_matrix);
    
end

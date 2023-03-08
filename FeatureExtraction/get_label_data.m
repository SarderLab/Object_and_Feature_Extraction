function label_matrix=get_label_data(excel_file,case_dir,classname,case_name_col,data_range,items_per_case)

[~,txt,~]=xlsread(excel_file,[case_name_col,':',case_name_col]);
[num,~,~]=xlsread(excel_file,[data_range{1},':',data_range{2}]);
biopsy_cases=dir(case_dir);
biopsy_cases(1:2)=[];
dirFlags=[biopsy_cases.isdir];
biopsy_cases=biopsy_cases(dirFlags);


excel_names={txt{:,1}};
label_matrix=[];
for i=1:numel(biopsy_cases)
    case_name=biopsy_cases(i).name;
    save_labels=[];
    if any(strcmp(excel_names,case_name))
        Idx=find(strcmp(excel_names,case_name));
        disp(Idx);
        segment_out_dir=fullfile(case_dir,case_name,classname,'/CompartmentSegmentations');
        seg_files=dir([segment_out_dir,'/*.png']);
        save_labels=[save_labels,i*ones(items_per_case(i,1),1)];   
        for j =1:size(num,2)
            save_labels=[save_labels,num(Idx,j)*ones(items_per_case(i,1),1)];
        end
    
    else
        'Mismatched case directory or excel entry'
        %print(case_name);
        save_labels = zeros(items_per_case(i,1),size(num,2)+1)-1;
    end
    label_matrix=[label_matrix;save_labels];
end
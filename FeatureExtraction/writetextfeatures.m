function writetextfeatures(txt,ordertxt,b_c,image_dir,features)

[d1,d2]=size(features);
for i=1:d1
    fprintf(ordertxt,[num2str(b_c),',',num2str(i),',',image_dir(i).name,'\n']);
     for j=1:d2
        fprintf(txt,[num2str(features(i,j)),',']);
     end
     fprintf(txt,'\n');
    if i==d1
     fprintf(txt,'---\n');
     fprintf(ordertxt,'---\n');
    end
end

%     [d1,d2]=size(features);
%     for i=1:d1
%         fprintf(ordertxt,[num2str(b_c),',',num2str(i),',',image_dir(i).name,'\n']);
%          for j=1:d2
%             fprintf(txt,[num2str(features(i,j)),',']);
%          end
%          fprintf(txt,'\n');
%         if i==d1
%          fprintf(txt,'---\n');
%          fprintf(ordertxt,'-b--\n');
%         end
%     end
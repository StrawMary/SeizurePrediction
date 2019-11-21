%% Split the segments into smaller chunks

% Preprocess data

clear
clc

patient = 'Pat1';

data_dir = ['..\contest_data_downloader\test_download\' patient 'Train\'];
output_dir = 'preprocessed\';

files = dir(data_dir);
preprocessed_data = [];
bad_patients = cell(0);

for filenum = 3:length(files)
    
    
    load([data_dir files(filenum).name]);
    filename = files(filenum).name;
    if min(std(data)) < .0001 % dropout data
        bad_patients{end+1} = filename;
        continue 
    end
    
    Fs = 400;
    
    data_full = data;
    
    disp('');
    for i = 0:19
        data = data_full(i*Fs*30+1:(i+1)*Fs*30,:);
        namesplit = strsplit(filename, '_');
        new_filename = sprintf('%s_%s_split%d_%s', namesplit{1}, namesplit{2}, i, namesplit{3});
        save(['C:\MLSP\contest_data_downloader\30sec_data\' new_filename], 'data');
    end 
    fprintf('finished %d / %d\n', filenum-2, length(files)-2);
    
end

%save([output_dir 'preprocessed_data2.mat'], 'preprocessed_data');




%Check that you are in the correct path: it should be the path ending with
%ODS OIS F-score Scripts.

addpath(pwd)
addpath(pwd, 'edges')
addpath(genpath(fullfile(pwd, 'toolbox')))

dataset_version = 'Combined Dataset Example'; %This is the name of the dataset version which is being tested.
network_version = 'Fine Tuned 4'; %This is the name of the fine tuning version which is being tested.
mixed_data_bool = true; %true;

%Preprocessing the ground truths into the correct format for the toolbox.
if mixed_data_bool == false
    if(~exist(fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real'),'dir')), mkdir((fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real'))); end
    
    ids=dir(fullfile(fullfile(pwd, 'data/Test Dataset GroundTruths Raw', dataset_version, 'test/rgbr/real'),'*.mat')); ids={ids.name}; n=length(ids);
    for i=1:n, id=ids{i};
        gt_im_path = fullfile(fullfile(pwd, 'data/Test Dataset GroundTruths Raw', dataset_version, 'test/rgbr/real'),[id]);
        loaded_image = load(gt_im_path).arr;
        gt_struct = struct('Boundaries', loaded_image);
        groundTruth = {gt_struct};
        output_path = fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real', id);
        save(output_path, 'groundTruth')
    end
else
    if(~exist(fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real/combined_real'),'dir')), mkdir((fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real/combined_real'))); end
    
    ids=dir(fullfile(fullfile(pwd, 'data/Test Dataset GroundTruths Raw', dataset_version, 'test/rgbr/real/combined_real'),'*.mat')); ids={ids.name}; n=length(ids);
    for i=1:n, id=ids{i};
        gt_im_path = fullfile(fullfile(pwd, 'data/Test Dataset GroundTruths Raw', dataset_version, 'test/rgbr/real/combined_real'),[id]);
        loaded_image = load(gt_im_path).arr;
        gt_struct = struct('Boundaries', loaded_image);
        groundTruth = {gt_struct};
        output_path = fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real/combined_real', id);
        save(output_path, 'groundTruth')
    end
end

if mixed_data_bool == true
   path_edgemap = fullfile(pwd, 'result', network_version, dataset_version, 'fused');
   path_gt = fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real/combined_real');
else
   path_edgemap = fullfile(pwd, 'result', network_version, dataset_version, 'fused');
   path_gt = fullfile(pwd, 'data/Test Dataset GroundTruths Processed', dataset_version, 'test/rgbr/real');
end

%path_edgemap = fullfile(pwd, "result");
%path_gt = fullfile(pwd, "data/Test Dataset GroundTruths Raw");

%imshow(arr)

[ODS,~,~,~,OIS,~,~,AP,R50] = edgesEvalDir({'resDir', path_edgemap, 'gtDir', path_gt}); %path_edgemap, path_gt);

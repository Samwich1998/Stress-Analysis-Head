%##########################################################################
%                           s01_extractViddata.m
%
%--------------------------------------------------------------------------
% Summary:
% csv/txt format files containing video duration and sequence information 
% are loaded and converted into numerical mat files. This is done, cause:
%   a) It is easier/faster in MATLAB to work with mat files.
%   b) The converted files are required for labeling in the later scripts.
%
%--------------------------------------------------------------------------
% The videos are given the following numbers to replace the string names:
% 
% startVid = 10
% bluVid = 11
% endVid = 12 
%
% amusing-1 = 1
% amusing-2 = 2
% boring-1  = 3
% boring-2  = 4
% relaxed-1 = 5
% relaxed-2 = 6
% scary-1   = 7
% scary-2   = 8
%--------------------------------------------------------------------------
%##########################################################################


%---------------------------Setup------------------------------------------

%$ empty workspace
clear all;

%$ Navigate to the case_dataset root directory
%$ NOTE: please set this up for your case e.g.
%$ cd /home/<USER>/Documents/case_Dataset   
cd /Users/karan/Documents/pewo/affectComp/case_dataset


%-------videos_duration.txt converted to num version: videos_duration_num.mat-------

%$ loading videos_duration.txt, the new way
in_vidsDuration = readtable('./metadata/videos_duration.txt', 'Delimiter', '\t');

%$ nrows of the table
in_vidsDuration_nrow = height(in_vidsDuration);
in_vidsDuration_ncol = width(in_vidsDuration);

%$ initializing empty table
vidsDuration = zeros(in_vidsDuration_nrow,in_vidsDuration_ncol);

%$ looping through column and each row to replace strings with numbers
for i= 1:1:in_vidsDuration_nrow
    
    
    vid2match = cellstr(in_vidsDuration.video_name{i});
    
    if strcmp(vid2match,'startVid') == 1 %startVid=10
        vidsDuration(i,1) = 10;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'bluVid') == 1 %bluVid=11 
        vidsDuration(i,1) = 11;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'endVid') == 1 %endVid=12
        vidsDuration(i,1) = 12;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'amusing-1') == 1 %amusing-1=1
        vidsDuration(i,1) = 1;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'amusing-2') == 1 %amusing-2=2
        vidsDuration(i,1) = 2;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'boring-1') == 1 %boring-1=3
        vidsDuration(i,1) = 3;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'boring-2') == 1 %boring-2=4
        vidsDuration(i,1) = 4;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'relaxed-1') == 1 %relaxed-1=5
        vidsDuration(i,1) = 5;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'relaxed-2') == 1 %relaxed-2=6
        vidsDuration(i,1) = 6;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'scary-1') == 1 %scary-1=7
        vidsDuration(i,1) = 7;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
        
    elseif strcmp(vid2match,'scary-2') == 1 %scary-2=8
        vidsDuration(i,1) = 8;
        vidsDuration(i,2) = table2array(in_vidsDuration(i,2));
    end        
end

%$ saving the new matrix as a mat file
save('./metadata/videos_duration_num.mat', 'vidsDuration')

%$ cleaning up
clear_varlist={'in_vidsDuration','in_vidsDuration_nrow',...
    'in_vidsDuration_ncol', 'i','vid2match'};
clear(clear_varlist{:});
clearvars clear_varlist;


%--------seqs_order.txt converted to num version: seqs_order_num.mat-----------

%$ loading seqs_order.txt, the new way
in_seqsOrder = readtable('./metadata/seqs_order.txt', 'Delimiter', '\t');

%$ dims of the table
in_seqsOrder_nrow = height(in_seqsOrder);
in_seqsOrder_ncol = width(in_seqsOrder);

%$ initializing empty table
seqsOrder = zeros(in_seqsOrder_nrow,in_seqsOrder_ncol);

%$ looping through column and each row to replace strings with numbers
for j= 1:1:in_seqsOrder_ncol
    
    for i = 1:1:in_seqsOrder_nrow
        vid2match = cellstr(in_seqsOrder{i,j});
        
        if strcmp(vid2match,'startVid') == 1 %startVid=10
            seqsOrder(i,j) = 10;
        elseif strcmp(vid2match,'bluVid') == 1 %bluVid=11
            seqsOrder(i,j) = 11;
        elseif strcmp(vid2match,'endVid') == 1 %endVid=12
            seqsOrder(i,j) = 12;
        elseif strcmp(vid2match,'amusing-1') == 1 %amusing-1=1
            seqsOrder(i,j) = 1;
        elseif strcmp(vid2match,'amusing-2') == 1 %amusing-2=2
            seqsOrder(i,j) = 2;
        elseif strcmp(vid2match,'boring-1') == 1 %boring-1=3
            seqsOrder(i,j) = 3;
        elseif strcmp(vid2match,'boring-2') == 1 %boring-2=4
            seqsOrder(i,j) = 4;
        elseif strcmp(vid2match,'relaxed-1') == 1 %relaxed-1=5
            seqsOrder(i,j) = 5;
        elseif strcmp(vid2match,'relaxed-2') == 1 %relaxed-2=6
            seqsOrder(i,j) = 6;
        elseif strcmp(vid2match,'scary-1') == 1 %scary-1=7
            seqsOrder(i,j) = 7;
        elseif strcmp(vid2match,'scary-2') == 1 %scary-2=8
            seqsOrder(i,j) = 8;
        end
        
    end
end

%$ saving the new matrix as a mat file
save('./metadata/seqs_order_num.mat', 'seqsOrder')

%$ cleaning up
clear_varlist={'in_seqsOrder','in_seqsOrder_nrow','in_seqsOrder_ncol','i', 'j',...
    'vid2match'};
clear(clear_varlist{:});
clear clear_varlist;



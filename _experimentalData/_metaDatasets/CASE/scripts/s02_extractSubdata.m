%##########################################################################
%                           s02_extractSubdata.m
%
%--------------------------------------------------------------------------
% Summary:
% Subject wise physiological signals and annotation data from raw txt data 
% files is extracted and converted into numerical mat files.
% This is done, cause:
%   a) Easier/faster in MATLAB to work with mat files.
%--------------------------------------------------------------------------
% Input: subject-wise raw annotation and physiological data files are
%   loaded from the data/raw sub-directory.
%
% Output: subject-wise mat files containing both annotations and
%   physiological data saved under data/intial sub-directory.
%--------------------------------------------------------------------------
%##########################################################################


%---------------------------Setup-----------------------------------------

%$ empty workspace
clear all;

%$ Navigate to the case_dataset root directory
%$ NOTE: please set this up for your case e.g.
%$ cd /home/<USER>/Documents/case_Dataset   
cd /Users/karan/Documents/pewo/affectComp/case_dataset

% verbose mode on (True) or off (False)
verbose = true;

%------------------------Loading and Extracting data-----------------------

% looping through all subjects
for subno = 1:1:30
    
    tmp_daqfilename = sprintf('./data/raw/physiological/sub%d_DAQ.txt',subno);
    tmp_jsfilename = sprintf('./data/raw/annotations/sub%d_joystick.txt',subno);
    
    tmp_daqdata = importdata(tmp_daqfilename);
    tmp_jsdata = importdata(tmp_jsfilename);
    
    [nrow_daqdata,ncol_daqdata] = size(tmp_daqdata);
    [nrow_jsdata,ncol_jsdata] = size(tmp_jsdata);
    
    % verbose quality check
    if verbose == true
       fprintf('Sub %d nrows in loaded rawdata for daqdata = %d and jsdata = %d \n',...
           subno,nrow_daqdata,nrow_jsdata);
    end
     
    %------------DAQ data-------------
    %$ extracting data
    daqtime = tmp_daqdata(:,1);
    ecg = tmp_daqdata(:,2);
    bvp = tmp_daqdata(:,3);
    gsr = tmp_daqdata(:,4);
    rsp = tmp_daqdata(:,5);
    skt = tmp_daqdata(:,6);
    emg_zygo = tmp_daqdata(:,7);
    emg_coru = tmp_daqdata(:,8);
    emg_trap = tmp_daqdata(:,9);
    
    %$ cleaning up
    clear_varlist1 = {'tmp_daqfilename', 'tmp_daqdata', 'nrow_daqdata',...
        'ncol_daqdata',};
    clear(clear_varlist1{:});
    clearvars clear_varlist1;
    
    %-------------Joystick data-------------
    %$ extracting data
    jstime = tmp_jsdata(:,1);
    val = tmp_jsdata(:,2);
    aro = tmp_jsdata(:,3);
    
    %$ cleaning up
    clear_varlist2 = {'tmp_jsfilename', 'tmp_jsdata', 'nrow_jsdata',...
        'ncol_jsdata', 'cols2remove'};
    clear(clear_varlist2{:});
    clearvars clear_varlist2;
    
    %------------Saving data-------------
    %$ list of variables to save in per subject mat file
    vars2save = {'daqtime','ecg', 'bvp', 'gsr', 'rsp' 'skt', 'emg_zygo',...
        'emg_coru', 'emg_trap', 'jstime', 'val', 'aro'};
    
    %$ saving data to a file
    tmp_outfilename = sprintf('./data/initial/sub_%d.mat',subno);
    save(tmp_outfilename,vars2save{:});
    
    %$ cleaning up
    clear_varlist3 = {'daqtime','ecg', 'bvp', 'gsr', 'rsp' 'skt', 'emg_zygo',...
        'emg_coru', 'emg_trap', 'jstime', 'val', 'aro',...
        'vars2save','tmp_outfilename'};
    clear(clear_varlist3{:});
    clearvars clear_varlist3;
end
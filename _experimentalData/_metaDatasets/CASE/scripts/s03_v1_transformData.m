%##########################################################################
%                         s03_v1_transformData.m
%
%--------------------------------------------------------------------------
% Summary: 
% The individual mat files containing the raw annotation and physiological
% data are loaded and the data containing within these files is manipulated
% as follows:
% 1. Load the individual physiological and annotation data from the mat
%    files, and apply the following conversions:
%    a. time is converted from seconds to ms (both physiological and
%       annotation data).
%    b. conversions from voltages to units for different sensors, these are
%       done on basis of conversions in thought technology datasheets.
%    c. annotation X and Y data are changed to range [0.5 9.5].
% 2. generate the video labels for the annotation and physiological data
%    using the f_labelData function.
% 3. add the video labels information to the manipulated data and save them
%    into different comma-separated-value (csv) files for annotation and
%    physiological data.
%--------------------------------------------------------------------------
%##########################################################################

%---------------------------Setup------------------------------------------

%$ empty workspace
clear all;

%$ Navigate to the case_dataset root directory
%$ NOTE: please set this up for your case e.g.
%$ cd /home/<USER>/Documents/case_Dataset   
cd /Users/karan/Documents/pewo/affectComp/case_dataset

% load files for labelling created in s01_extractViddata.m
%$ loading variable vidsDuration from vids_dur_num.mat
load('./metadata/videos_duration_num.mat');
%$ loading varialbe seqsOrder from seqs_order_num.mat
load('./metadata/seqs_order_num.mat');

% verbose mode on (True) or off (False)
verbose = true;

% declare function for rounding to 3 decimal places
round3func = @(DT) round(DT, 3); % DT is datatable

% source the f_labelData function to be used here
addpath('./scripts/')

%----------------------Loading and extracting data-------------------------
% looping through all subjects
for subno = 1:1:30
    
    % temp file name and load data
    indata_filename = sprintf('./data/initial/sub_%d.mat',subno);
    indata = load(indata_filename);
    
    % verbose quality check
    if verbose == true
       fprintf('Sub %d nrows in initial matfiles for daqdata = %d and jsdata = %d \n',...
           subno,length(indata.daqtime),length(indata.jstime));
    end
    
    %-------------------------Transforming Data----------------------------
    
    %------DAQ data------
    
    % daqtime -- conv. from seconds to milliseconds
    tmp_daqtime = indata.daqtime * 1000;
    
    % ECG -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: usually measured in millivolts (sensor I/P range +-40 mV)
    tmp_ecg = (indata.ecg - 2.8) / 50;
    %$ converting volts to milliVolts (mV) & rounding to 3 places
    tmp_ecg = (tmp_ecg * 1000);
    
    % BVP -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: BVP% (range 0-100%)
    tmp_bvp = (58.962 * indata.bvp) - 115.09;
    
    % GSR -- conv. O/P voltage(acq on DAQ) to I/P
    % I/P extracted: conductance, measured in uSiemens (sensor I/P range +-40 mV)
    tmp_gsr = (24 * indata.gsr) - 49.2;
    
    % Respiration -- conv. O/P voltage(acq on DAQ) to I/P
    % I/P extracted: sensor elongation in % (simply respiration%)
    tmp_rsp = (58.923 * indata.rsp) - 115.01;
    
    % Skin tmperature -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: temp (degree Celsius)
    tmp_skt = (21.341 * indata.skt) - 32.085;
    
    % EMG Zygo, Corru and Trap -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: measured I/P RMS voltage, measured in uV (sensor I/P
    %range 0-400 uV_rms)
    tmp_zygo = (indata.emg_zygo - 2.0) / 4000;
    tmp_coru = (indata.emg_coru - 2.0) / 4000;
    tmp_trap = (indata.emg_trap - 2.0) / 4000;
    %$ converting volts to microVolts (uV)
    tmp_zygo = tmp_zygo * 1000000;
    tmp_coru = tmp_coru * 1000000;
    tmp_trap = tmp_trap * 1000000;
    
    %------Joystick data------
    
    %joystick time -- conv. from seconds to milliseconds
    tmp_jstime = indata.jstime * 1000;
    
    % valence(x) and arousal(y) converting to range [0.5 9.5]
    tmp_val = 0.5+9*(indata.val+26225)/52450;
    tmp_aro = 0.5+9*(indata.aro+26225)/52450;
    
    %-------Cleaning up------
    clearvars indata indata_filename;
    
    
    %---------------------------Labeling the data--------------------------
    
    % extracting the sequence for current subject
    csub_vidseq = seqsOrder(:,subno);
    
    %------DAQ data------
    
    [csub_daqlabels] = f_labelData(tmp_daqtime,csub_vidseq,vidsDuration,...
        subno);
    
    % combining daqdata and label vector into a data.table
    %-------------------------------
    % The columns are ordered:
    % | time(ms) | ECG | BVP | GSR | RESP | SkinTemp | EMG-Zygo | ...
    % EMG-Corru | EMG-Trap | video |
    %-------------------------------
    csub_daqtable = table(tmp_daqtime, tmp_ecg, tmp_bvp, tmp_gsr,...
        tmp_rsp, tmp_skt, tmp_zygo, tmp_coru, tmp_trap, csub_daqlabels);
  
    csub_daqtable.Properties.VariableNames = {'daqtime' 'ecg' 'bvp' 'gsr'...
        'rsp' 'skt' 'emg_zygo' 'emg_coru' 'emg_trap' 'video'};
    
    % Trimming csub_daqtable
    %-------------------------------
    % During labeling the rows which exceed the total cumaltive duration
    % of the sequence have label = 0 (as initialized). These extra rows
    % are trimmed.
    %-------------------------------
    
    % which rows are to be deleted
    labelsdaq_nrow = length(csub_daqlabels);
    rows_2b_deleted = find(csub_daqlabels == 0);
    
    % removing the rows from the complete daqdata table
    csub_daqtable(rows_2b_deleted,:) = [];
    
    % verbose quality check
    if verbose == true
       fprintf('Sub %d nrows for trimmed csub_daqtable= %d and no. of trimmed rows = %d \n',...
           subno,length(csub_daqtable.daqtime),length(rows_2b_deleted));
    end
    
    % cleaning up
    clear_varlist1 = {'tmp_daqtime','tmp_ecg','tmp_bvp','tmp_gsr',...
        'tmp_rsp','tmp_skt','tmp_zygo','tmp_coru','tmp_trap',...
        'csub_daqlabels'};
    clear_varlist2 = {'labelsdaq_nrow', 'rows_2b_deleted'};
    clear(clear_varlist1{:});
    clear(clear_varlist2{:});
    clearvars clear_varlist1 clear_varlist2;
    
    %------Joystick data------
    
    % creating the label vector for joystick data
    [csub_jslabels] = f_labelData(tmp_jstime,csub_vidseq,...
        vidsDuration,subno);
    
    % combining joystick data and label vector into a data.table
    %-------------------------------
    % The columns are ordered:
    % | time(ms) (joystick time) | valence(x) | arousal (y) | video |
    %-------------------------------
    csub_jstable = table(tmp_jstime, tmp_val, tmp_aro, csub_jslabels);
    
    % renaming the columns of the table
    csub_jstable.Properties.VariableNames = {'jstime' 'valence' 'arousal'...
        'video'};
    
    % Trimming csub_jstable
    %-------------------------------
    % During labeling the rows which exceed the total cumaltive duration
    % of the sequence have label = 0 (as initialized). These extra rows
    % are trimmed.
    %-------------------------------
    
    % which rows are to be deleted
    labelsjs_nrow = length(csub_jslabels);
    rows_2b_deleted = find(csub_jslabels == 0);
    
    % removing the rows from the complete jsdata table
    csub_jstable(rows_2b_deleted,:) = [];
    
    % verbose quality check
    if verbose == true
       fprintf('Sub %d nrows for trimmed csub_jstable= %d and no. of trimmed rows = %d \n',...
           subno,length(csub_jstable.jstime),length(rows_2b_deleted));
    end
    
    % cleaning up
    clear_varlist1 = {'tmp_jstime', 'tmp_val', 'tmp_aro', 'csub_jslabels'};
    clear_varlist2 = {'labelsjs_nrow', 'rows_2b_deleted'};
    clear(clear_varlist1{:});
    clear(clear_varlist2{:});
    clearvars clear_varlist1 clear_varlist2;
    
    
    %-----------------Saving data for the csub-----------------------------
     
    %---------DAQ data-------
    % Rounding the data to 3 decimal places
    %$ round the variables and store into a new datatable
    mod_csub_daqDT = varfun(round3func,csub_daqtable);
    %$ reset variable names that changed due to varfun operation
    daq_varnames = csub_daqtable.Properties.VariableNames;
    mod_csub_daqDT.Properties.VariableNames = daq_varnames;
    
    % cleanup
    clearvars csub_daqtable daq_varnames;
    
    % save data
    tmp_outfileid_daq = sprintf('./data/non-interpolated/physiological/sub_%d.csv',subno);
    writetable(mod_csub_daqDT,tmp_outfileid_daq);
    
    % cleanup
    clearvars mod_csub_daqDT tmp_outfileid_daq;
    
    %---------Annotation data-------
    % Rounding the data to 3 decimal places
    %$ round the variables and store into a new datatable
    mod_csub_jsDT = varfun(round3func,csub_jstable);
    %$ reset variable names that changed due to varfun operation
    js_varnames = csub_jstable.Properties.VariableNames;
    mod_csub_jsDT.Properties.VariableNames = js_varnames;
    
    % cleanup
    clearvars csub_jstable js_varnames;
    
    % save data
    tmp_outfileid_js = sprintf('./data/non-interpolated/annotations/sub_%d.csv',subno);
    writetable(mod_csub_jsDT,tmp_outfileid_js);
    
    % cleanup
    clearvars mod_csub_jsDT tmp_outfileid_js;
    
end

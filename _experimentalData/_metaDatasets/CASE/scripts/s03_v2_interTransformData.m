%##########################################################################
%                         s03_v2_interTransformData.m
%
%--------------------------------------------------------------------------
% Summary:
% The individual mat files containing the raw annotation and physiological
% data are loaded and the data containing within these files is manipulated
% as follows:
% 1. Load the individual physiological and annotation data from the mat
%    files, and apply the following initial conversion:
%    a. time is converted from seconds to ms (both physiological and
%       annotation data).
% 2. Inter-/Extra-polate data: based on the pre-determined duration of a
%    complete video sequence (i.e. the sequence containing the emotion and
%    blue videos) as viewed by each participant. Since the videos used for
%    all participants are the same, with the difference being in their
%    ordering, the duration of the sequence was same across all participants.
%    It was 2451583.333 milliseconds.
% 3. Transfrom data:
%    a. conversions from voltages to units for different sensors, these are
%       done on basis of conversions in thought technology datasheets.
%    b. annotation X and Y data are changed to range [0.5 9.5].
% 4. generate the video labels for the annotation and physiological data
%    using the f_labelData function.
% 5. add the video labels information to the manipulated data and save them
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
cd ../

% load files for labelling created in s01_extractViddata.m
%$ loading variable vidsDuration from vids_dur_num.mat
load('./metadata/videos_duration_num.mat');
%$ loading varialbe seqsOrder from seqs_order_num.mat
load('./metadata/seqs_order_num.mat');

% verbose mode on (True) or off (False)
verbose = true;

% plotting mode on (True) or off (False)
plotting = false;

% declare function for rounding to 3 decimal places
round3func = @(DT) round(DT, 3); % DT is datatable

% source the f_labelData function to be used here
addpath('./scripts/')

% duration of a complete video sequence in milliseconds
% CAUTION: do not edit the following variable
VID_SEQ_DUR = 2451583.333;

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
   
    %------DAQ data------
    tmp_daqtime = indata.daqtime * 1000; %conv. from sec. to milliseconds
    tmp_ecg = indata.ecg;
    tmp_bvp = indata.bvp;
    tmp_gsr = indata.gsr;
    tmp_rsp = indata.rsp;
    tmp_skt = indata.skt;
    tmp_zygo = indata.emg_zygo;
    tmp_coru = indata.emg_coru;
    tmp_trap = indata.emg_trap;
    %$ combined daq data without time column
    tmp_daqdata = [tmp_ecg, tmp_bvp, tmp_gsr, tmp_rsp, tmp_skt, ...
        tmp_zygo, tmp_coru, tmp_trap];
    
    %------Joystick data------
    tmp_jstime = indata.jstime * 1000; %conv. from sec. to milliseconds
    tmp_val = indata.val;
    tmp_aro = indata.aro;
    %$ combined js data without time column
    tmp_jsdata = [tmp_val, tmp_aro];
    
    %-------Cleaning up------
    clearvars indata indata_filename;
    
    
    %------------------------Inter-/Extra-polation-------------------------
    
    %-------inter-/extra-polation time vectors------
    % standard time for all sequnces to inter/extrapolate to
    inter_daqtime = (0:1:VID_SEQ_DUR)'; % sample-time: 1 ms
    inter_jstime = (0:50:VID_SEQ_DUR)'; % sample-time: 50 ms
    
    %-------interpolation (also extropolate if necessary)-------
    %$ daq data
    inter_daqdata = interp1(tmp_daqtime, tmp_daqdata, inter_daqtime,...
        'linear', 'extrap');
    %$ joystick
    inter_jsdata = interp1(tmp_jstime, tmp_jsdata, inter_jstime,...
        'linear', 'extrap');
    
    % verbose quality check
    if verbose == true
        fprintf('Sub %d nrows in tmp_daqtime = %d, inter_daqtime = %d and inter_daqdata = %d \n',...
            subno,length(tmp_daqtime),length(inter_daqtime),length(inter_daqdata));
        fprintf('Sub %d nrows in tmp_jstime = %d, inter_jstime = %d and inter_jsdata = %d \n',...
            subno,length(tmp_jstime),length(inter_jstime),length(inter_jsdata));
    end
    
    % plot non-interpolated and interpolated data
    if plotting == true
        
        %$ DAQ data
        %$$ ECG, BVP and GSR
        figure('Name', 'Comparison of ECG, BVP and GSR');
        subplot(3,1,1);
        plot(tmp_daqtime,tmp_daqdata(:,1),':r',...
            inter_daqtime,inter_daqdata(:,1),':k');
        subplot(3,1,2);
        plot(tmp_daqtime,tmp_daqdata(:,2),':r',...
            inter_daqtime,inter_daqdata(:,2),':k');
        subplot(3,1,3);
        plot(tmp_daqtime,tmp_daqdata(:,3),':r',...
            inter_daqtime,inter_daqdata(:,3),':k');
        %$$ Respiration and Skin-Temp
        figure('Name', 'Comparison of RSP and SKT');
        subplot(2,1,1);
        plot(tmp_daqtime,tmp_daqdata(:,4),':r',...
            inter_daqtime,inter_daqdata(:,4),':k');
        subplot(2,1,2);
        plot(tmp_daqtime,tmp_daqdata(:,5),':r',...
            inter_daqtime,inter_daqdata(:,5),':k');
        %$$ Electromyography data
        figure('Name', 'Comparison of EMG- ZYGO, CORU and TRAP');
        subplot(3,1,1);
        plot(tmp_daqtime,tmp_daqdata(:,6),':r',...
            inter_daqtime,inter_daqdata(:,6),':k');
        subplot(3,1,2);
        plot(tmp_daqtime,tmp_daqdata(:,7),':r',...
            inter_daqtime,inter_daqdata(:,7),':k');
        subplot(3,1,3);
        plot(tmp_daqtime,tmp_daqdata(:,8),':r',...
            inter_daqtime,inter_daqdata(:,8),':k');
        
        %$$ Joystick data
        figure('Name', 'Comparison of Joystick Data - VAL and ARO');
        subplot(2,1,1);
        plot(tmp_jstime,tmp_jsdata(:,1),':r',...
            inter_jstime,inter_jsdata(:,1),':k');
        subplot(2,1,2);
        plot(tmp_jstime,tmp_jsdata(:,2),':r',...
            inter_jstime,inter_jsdata(:,2),':k');
    end
    
    %-------Cleaning up-----------
    % removing temporary non-interploated data
    clear_varlist1 = {'tmp_daqtime', 'tmp_ecg', 'tmp_bvp', 'tmp_gsr',...
        'tmp_rsp', 'tmp_skt', 'tmp_zygo', 'tmp_coru' 'tmp_trap'};
    clear_varlist2 = {'tmp_jstime', 'tmp_val', 'tmp_aro'};
    clear_varlist3 = {'tmp_daqdata', 'tmp_jsdata'};
    clear(clear_varlist1{:});
    clear(clear_varlist2{:});
    clear(clear_varlist3{:});
    clearvars clear_varlist1 clear_varlist2 clear_varlist3;
    
    
    %-------------------------Transforming Data----------------------------
    
    %------DAQ data------
    
    % Time -- inter_daqtime doesn't need to be trasformed (is already in ms)
    daqtime = inter_daqtime;
    
    % ECG -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: usually measured in millivolts (sensor I/P range +-40 mV)
    ecg = (inter_daqdata(:,1) - 2.8) / 50;
    %$ converting volts to milliVolts (mV)
    ecg = (ecg * 1000);
    
    % BVP -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: BVP% (range 0-100%)
    bvp = (58.962 * inter_daqdata(:,2)) - 115.09;
    
    % GSR -- conv. O/P voltage(acq on DAQ) to I/P
    % I/P extracted: conductance, measured in uSiemens (sensor I/P range +-40 mV)
    gsr = (24 * inter_daqdata(:,3)) - 49.2;
    
    % Respiration -- conv. O/P voltage(acq on DAQ) to I/P
    % I/P extracted: sensor elongation in % (simply respiration%)
    rsp = (58.923 * inter_daqdata(:,4)) - 115.01;
    
    % Skin tmperature -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: temp (degree Celsius)
    skt = (21.341 * inter_daqdata(:,5)) - 32.085;
    
    % EMG Zygo, Corru and Trap -- conv. O/P voltage(acquired on DAQ) to I/P
    % I/P extracted: measured I/P RMS voltage, measured in uV (sensor I/P
    % range 0-400 uV_rms
    emg_zygo = (inter_daqdata(:,6) - 2.0) / 4000;
    emg_coru = (inter_daqdata(:,7) - 2.0) / 4000;
    emg_trap = (inter_daqdata(:,8) - 2.0) / 4000;
    %$ converting volts to microVolts (uV)
    emg_zygo = emg_zygo * 1000000;
    emg_coru = emg_coru * 1000000;
    emg_trap = emg_trap * 1000000;
    
    %------Joystick data------
    
    % Time -- inter_jstime doesn't need to be trasformed (is already in ms)
    jstime = inter_jstime;
    
    % Valence(x) and Arousal(y) converting to range [0.5 9.5]
    val = 0.5+9*(inter_jsdata(:,1)+26225)/52450;
    aro = 0.5+9*(inter_jsdata(:,2)+26225)/52450;
    
    %-----Cleaning up-------
    clear_varlist1 = {'inter_daqtime', 'inter_daqdata',...
        'inter_jstime', 'inter_jsdata'};
    clear(clear_varlist1{:});
    clearvars clear_varlist1;
    
    
    %---------------------------Labeling the data--------------------------
    
    % extracting the sequence for current subject
    csub_vidseq = seqsOrder(:,subno);
    
    %------DAQ data------
    
    % Creating the label vector for daq data
    [csub_daqlabels] = f_labelData(daqtime,csub_vidseq,vidsDuration,subno);
    
    % combining daqdata and label vector into a data.table
    %-------------------------------
    % The columns are ordered:
    % | time(ms) | ECG | BVP | GSR | RESP | SkinTemp | EMG-Zygo | ...
    % EMG-Corru | EMG-Trap | video |
    %-------------------------------
    csub_daqtable = table(daqtime, ecg, bvp, gsr, rsp, skt,...
        emg_zygo, emg_coru, emg_trap, csub_daqlabels);
    
    csub_daqtable.Properties.VariableNames = {'daqtime' 'ecg' 'bvp' 'gsr' ...
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
    clear_varlist1 = {'daqtime','ecg','bvp','gsr','rsp','skt',...
        'emg_zygo','emg_coru','emg_trap', 'csub_daqlabels'};
    clear_varlist2 = {'labelsdaq_nrow', 'rows_2b_deleted'};
    clear(clear_varlist1{:});
    clear(clear_varlist2{:});
    clearvars clear_varlist1 clear_varlist2;
    
    %------Joystick data------
    
    % creating the label vector for joystick data
    [csub_jslabels] = f_labelData(jstime,csub_vidseq,vidsDuration,subno);
    
    % combining joystick data and label vector into a data.table
    %-------------------------------
    % The columns are ordered:
    % | time(ms) (joystick time) | valence(x) | arousal (y) | video |
    %-------------------------------
    csub_jstable = table(jstime, val, aro, csub_jslabels);
    
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
    clear_varlist1 = {'jstime', 'val', 'aro', 'csub_jslabels'};
    clear_varlist2 =  {'labelsjs_nrow', 'rows_2b_deleted'};
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
    tmp_outfileid_daq = sprintf('./data/interpolated/physiological/sub_%d.csv',subno);
    writetable(mod_csub_daqDT,tmp_outfileid_daq);
    
    % cleanup
    clearvars mod_csub_daqDT tmp_outfileid_daq;
    
    %---------Annotation data-------
    %$ round the variables and store into a new datatable
    mod_csub_jsDT = varfun(round3func,csub_jstable);
    %$ reset variable names that changed due to varfun operation
    js_varnames = csub_jstable.Properties.VariableNames;
    mod_csub_jsDT.Properties.VariableNames = js_varnames;
    
    % cleanup
    clearvars csub_jstable js_varnames;
    
    % save data
    tmp_outfileid_js = sprintf('./data/interpolated/annotations/sub_%d.csv',subno);
    writetable(mod_csub_jsDT,tmp_outfileid_js);
    
    % cleanup
    clearvars mod_csub_jsDT tmp_outfileid_js;

end


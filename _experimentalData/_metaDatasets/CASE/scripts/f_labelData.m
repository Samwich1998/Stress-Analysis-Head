%##########################################################################
%                           f_labelData.m
%
%--------------------------------------------------------------------------
% Summary:
% This function is a helper function that returns a labelled vector for the
% inputed data. This function operates on a per subject level and returns
% a label vector that is later added to the intial data for further
% processing.
%--------------------------------------------------------------------------
%##########################################################################

function [labelvec] = f_labelData(timevec,vidseq_order,vidDur,sub)

%-----------------------------Setup----------------------------------------

% number of rows in timevec
timevec_nrow = length(timevec);

% creating an empty column to hold labels
csub_label = zeros(timevec_nrow,1);
    
% number of videos (rows) and no of subjects(cols) of seqsOrder
n_vidsinSeq = length(vidseq_order);

% set that sequence loop is not initialised
csub_seqloop = 0;

for j = 1:1:n_vidsinSeq %looping through vidoes in a seq
    
    if csub_seqloop == 0
        last_time_found = 0;
        video_pos_start = 1;
        video_time_start = 0;
    elseif csub_seqloop == 1
        last_time_found = video_time_end;
        video_pos_start = video_pos_end + 1;
        video_time_start = timevec(video_pos_start,1);
    end
    
    % getting the j-th video in sequence for current subject
    vid_searched = vidseq_order(j);
    % finding the position of the j-video in vidDur table
    row_in_vidDur = find(vid_searched == vidDur(:,1));
    % getting the name and duration for the matched video
    vid_matched = vidDur(row_in_vidDur,1);
    vid_matched_dur = vidDur(row_in_vidDur,2); % for duration information
    
    % check -- throw error if wrong
    if vid_searched ~= vid_matched
        error('Something is wrong: vid_searched != vid_matched');
    end
    
    % time to be matched with the time column of timevec
    time2match = last_time_found + vid_matched_dur;
    
    % making a temporary time vector to search in for closest match
    tmp_timeVec = abs(timevec - time2match);
    
    % finding closest time & location of the of time2match timevec
    loc_time2match = find(tmp_timeVec == min(tmp_timeVec),1); %return 1st match
    closestTime_time2match = timevec(loc_time2match);
    
    % updating video_pos_end and video_time_end
    video_pos_end = loc_time2match;
    video_time_end = closestTime_time2match;
    
    % update csub_label with the video label
    csub_label(video_pos_start:video_pos_end) = vid_searched;
    
    % initialize loop sequence
    csub_seqloop = 1;
    
end

% label vector to be returned
labelvec = csub_label;

% clearing csub_label
clearvars csub_label;

end
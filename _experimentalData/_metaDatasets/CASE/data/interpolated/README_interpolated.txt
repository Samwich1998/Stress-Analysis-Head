%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%		README -- CASE_dataset/data/interpolated
%
% This short guide to the interpolated data, covers the following topics:
% (1) Preamble.
% (2) Need for interpolation.
% (3) Structure of this subfolder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------------------------------------------
(1) Preamble:
-------------------------------------------------------------------------------
The raw data acquired from each participant during the experiment is stored in
two different tab delimited text files. Where, one contains the physiological,
and the other, the annotation data. This was required because the the sampling
rates for the DAQ and annotation setups are different, i.e., 1000 Hz and 20 Hz,
respectively. Due to hardware restrictions, the sampling rate for annotation
joystick could not be set higher than 20 Hz.

-------------------------------------------------------------------------------
(2) Need for interpolation:
-------------------------------------------------------------------------------
Due to technical restrictions and logging delays, the samples might not always
be logged at a consistently pre-determined rate. For more details on the issue,
see the data descriptor. This issue can however be solved by inter/extra-
polating the data where necessary. This folder contains the data that was
interpolated using standard linear interpolation as implemented in the 
MATLAB/GNU-Octave function "interp1" (see accompanying scripts).

-------------------------------------------------------------------------------
(3) Structure of this subfolder:
-------------------------------------------------------------------------------
This subfolder to the dataset contains the following two subfolders that
respectively contain the physiological and annotation data for all 30
participants:

(a) CASE_dataset/data/interpolated/physiological
(b) CASE_dataset/data/interpolated/annotations

Each of above mentioned subfolders contain a README file explaining the data
present there.

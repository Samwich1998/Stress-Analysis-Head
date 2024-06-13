%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			README -- CASE_dataset/data
%
% This short guide to the data, covers the following topics:
% (1) Preamble.
% (2) Structure of this subfolder.
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

The acquired raw data needs to be pre-processed before any downstream analysis.
This folder and its sub-folders contain the various forms of data that can be
used for any such analyses. Detailed information on these data can be found in
the data descriptor. The following section provides a brief overview of the
structure of this sub-folder (and the subsequent sub-folders) and the data
contained in them.   

-------------------------------------------------------------------------------
(2) Structure of this subfolder:
-------------------------------------------------------------------------------
This subfolder to the dataset contains the following two subfolders that
respectively contain the physiological and annotation data for all 30
participants:

(a) CASE_dataset/data/raw - contains the raw data as acquired from LabVIEW,
	without any video-IDs. This folder is further sub-divided into:
	1. /annotations
	2. /physiological

(b) CASE_dataset/data/initial - contains mat files generated from raw data. A
	single mat file contains both annotation and physiological data.

(c) CASE_dataset/data/interpolated - contains data that has been pre-processed.
	This entails, e.g., addition of video-IDs, conversion of voltages to
	appropriate units. Also the data have been interpolated (see data
	descriptor). This folder is further sub-divided into:
	1. /annotations
	2. /physiological

(d) CASE_dataset/data/non-interpolated - contains data that have been 
	pre-processed (as above), with the difference being that they are not
	interpolated. This folder is further sub-divided into:
	1. /annotations
	2. /physiological

Each of above mentioned subfolders also contain a README file explaining the
data present there.

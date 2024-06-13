%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			README -- CASE_dataset/scripts
%
% This short guide to this dataset, covers the following topics:
% (1) General information.
% (2) Usage.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------------------------------------------
(1) General Information:
-------------------------------------------------------------------------------
This sub-folder contains the code (MATLAB scripts) that is used to process the
raw data into a form we deem suitable for use by others. Pre-processing the
acquired raw data is a multi-step process and has been appropriately split
across the following scripts:

	1. s01_extractVidData.m : the duration information for the different
	videos and the information about the sequence order of these videos for
	each subject are extracted from txt/csv files and converted to mat
	format for use by later scripts.

	2. s02_extractSubData.m : the raw physiological and annotation data for
	each subject are extracted from the respective txt files in the raw
	data folder and converted to a single mat format file that is saved in
	the initial data folder. 

	3. s03_v1_transformData.m : the individual mat files (in the initial
	data folder) containing the raw annotation and physiological data are
	loaded and the data containing within these files is manipulated as
	follows:
		(i) Load the individual physiological and annotation data from
		the mat files, and apply the following conversions:
			(a) Time is converted from seconds to ms (both
			physiological and annotation data).
    			(b) Conversions from voltages to units for different
			sensors, these are done on basis of conversions in
			thought technology datasheets.
			(c) Annotation X and Y data are changed to range
			[0.5 9.5].
		(ii) Generate the video labels (video-IDs) for the annotation
		and physiological data using the f_labelData function (more on
		this later).
		(iii) Add the video labels information to the manipulated data
		and save them into different comma-separated-value (csv) files
		for annotation and physiological data (see non-interpolated
		data folder).
	
	4. s03_v2_interTransformData.m : the individual mat files (in the 
	initial data folder) containing the raw annotation and physiological
	data are loaded and the data containing within these files is 
	manipulated as follows:
		(i) Load the individual physiological and annotation data 
		from the mat files, and apply the following initial conversion:
			(a) time is converted from seconds to ms (both
			physiological and annotation data).
		(ii) Inter-/Extra-polate data: based on the pre-determined
		duration of a complete video sequence (i.e. the sequence
		containing the emotion and blue videos) as viewed by each
		participant. Since the videos used for all participants are the
		same, with the difference being in their ordering, the duration
		of the sequence was same across all participants. It was
		2451583.333 milliseconds.
		(iii) Transfrom data:
			(a) conversions from voltages to units for different
			sensors, these are done on basis of conversions in
			thought technology datasheets.
			(b) annotation X and Y data are changed to range
			[0.5 9.5].
 		(iv) Generate the video labels for the annotation and
		physiological data using the f_labelData function (more on
		this later).
		(v) Add the video labels information to the manipulated data
		and save them into different comma-separated-value (csv) files
		for annotation and physiological data (see interpolated data
		folder).

	5. f_labelData.m : this helper function returns a labelled (video-ID) 
	vector for the inputed data. It operates on a per subject level and is
	used by s03_v1_transformData.m and s03_v2_intertransformData.m to
	determine the appropriate video-ID vector given the unique sequence for
	each subject. The returned vector from this function is later added to
	the initial data and is used for further processing.

-------------------------------------------------------------------------------
(2) Usage:
-------------------------------------------------------------------------------
- The MATLAB version used the pre-processing was 2014b (x64).

- All the scripts also contain comments to orient the user.

- Users who wish to replicate the data pre-processing step (i.e. generation of
  this dataset from raw data) should first change the file-paths in the scripts.
  They are clearly marked at the beginning of each script.

- Users should run the scripts in the appropriate order, which is highlighted
  in their titles (e.g. s01_extractVidData.m).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%			README -- CASE_dataset/metadata
%
% This short guide to the metadata folder, covers the following topics:
% (1) General Information.
% (2) Reasons for sequential ordering of videos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------------------------------------------
(1) General Information:
-------------------------------------------------------------------------------
The files in this folder are to a large extent, self-explanatory. Hence, only a
short description of each file is presented here:

	(1) participants.xlsx : is an excel file that contains some information
	on the participants for the study. This information can be used to
	categorize research results based on age-groups, sex, etc.

	(2) seqs_order.txt, seqs_order_num.mat and seqs_order_num.xlsx: contain
	the video sequences information for each participant. They contain the
	same data, just in different formats. Where,
		- seqs_order.txt: is the tab-delimited text file containing the
		sequence for each participant, where labels are used instead of
		IDs.
		- seqs_order_num.mat and seq_order_num.xlsx: during 
		pre-processing, the video-labels in seqs_order.txt file are
		replaced with video-IDs to reduce the size of the data files
		when these video-IDs are added to them (more on this later).
		Both these files contain the same data, with two different
		formats being presented for the convenience of the users.    
	
	(3) videos_duration.txt, videos_duration_num.mat and 
	videos_duration_num.xlsx: contain the duration in milliseconds of the
	different videos used during collection of this dataset. These videos
	are combined to make a unique sequence for each participant (see (2)).
	As was the case with the files mentioned in point (2), these files
	also contain the same data, just in different formats. Where
	videos_duration.txt uses video-labels and, videos_duration_num.mat and 		  		videos_duration_num.xlsx, use video-IDs.  

	(4) videos.xlsx: is an excel file that contains information on the
	videos used for the test. It contains the video-labels and video-IDs
	used to refer to these videos in the data descriptor and the dataset.
	The assigned video-IDs are the numerical equivalents to video-labels and
	are added to the dataset files during pre-processing. The reasons for
	using numerical equivalents instead of full video-names, are twofold:
		(i) in most scientific computing environments (e.g., MATLAB),
		it is much easier to handle numerical data over character
		strings.
		(ii) by using numerical video-IDs, file sizes can be kept
		smaller in comparison to when using character strings. Besides
		this, information pertaining to the videos' durations in
		milliseconds, URLs to the IMDb/YouTube entries for the videos'
		sources, URLs to the videos and their duration at these URLs,
		is presented. 

-------------------------------------------------------------------------------
(2) Reasons for sequential ordering of videos:
-------------------------------------------------------------------------------
To avoid carry-over effects in our within-subjects experiment design, a
unique ordering of the 8 emotional videos (e.g., video IDs: 1 to 8) for
each participant was generated pseudo-randomly, such that no two videos of the
same overall emotion-type follow each other in the sequence. These emotional
videos are interleaved by 2 minute long blue screens to better isolate their
effects. Two other videos (startVid and endVid) were also used in the experiment
and although the physiological and annotation data is also provided for these
videos, it is not integral to any subsequent analysis undertaken by the
researcher.

More information on the videos is available in the data descriptor and our
previous research.

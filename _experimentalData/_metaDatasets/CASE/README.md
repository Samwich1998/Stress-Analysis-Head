## README -- CASE_dataset

This short guide to this dataset, covers the following topics:

1. General information about the dataset.
2. Structure of the dataset.
3. Usage.

### 1. General Information:

The Continuously Annotated Signals of Emotion (CASE) dataset was developed to
address the lack of continuously and simultaneously annotated physiological
datasets for Affective Computing.

Accordingly, this dataset contains, (a) continuous Valence-Arousal (V-A)
annotations (done simultaneously in a 2-dimensional space) of emotional response
elicited using videos, and (b) measurements from several physiological sensors
acquired while the participants watched the videos and annotated their emotional
response.

The videos used for preparing this dataset cannot be shared due to copyright
restrictions. Nevertheless, further details about the same are provided in the
metadata folder, our previous research publications and in the data descriptor
accompanying this dataset. Please note however, that these videos are not
required to use this dataset, as the video-label information has been added to
the data during pre-processing.

More information on the dataset can be found in the accompanying data descriptor
and our previous (as of June, 2019) research work:

```
	(a) Sharma, K., Wagner, M., Castellini, C., van den Broek, E. L., Stulp,
	F. and Schwenker, F. (2019) A functional data analysis approach for
	continuous 2-D emotion annotations. Web Intelligence, 17 (1), pp. 41-52.
	IOS Press. DOI: 10.3233/web-190399 ISSN 2405-6456. 

	(b) Sharma, K., Castellini, C., Stulp, F.  and van den Broek, E. L. (2017)
	Continuous, real-time emotion annotation: A novel joystick-based analysis
	framework. IEEE Transactions on Affective Computing, PP (99), 1-1,
	DOI: 10.1109/TAFFC.2017.2772882 ISSN 1949-3045.
	
	(c) Sharma, K., Castellini, C. & van den Broek, E. L. (2016) Continuous
	affect state annotation using a joystick-based user interface: 
	Exploratory data analysis. In Measuring Behavior 2016: 10th 
	International Conference on Methods and Techniques in Behavioral
	Research, 500-505.
	
	(d) Antony, J., Sharma, K., van den Broek, E. L., Castellini, C. & 
	Borst, C. (2014) Continuous affect state annotation using a joystick-
	based user interface. In Proceedings of Measuring Behavior 2014:
	9th International Conference on Methods and Techniques in Behavioral
	Research, 268-271. 
``` 

### 2. Structure of the Dataset:

The root folder of the repository, where this README file is located, has the
following subfolders:

1. /data
2. /metadata
3. /scripts
 
Each of these subfolders and any further subfolders within them also contain
README files for more information. As such, only a short description of
these subfolders is provided here:

1. **/data**: contains several subfolders, that are often further subdivided into
    physiological and annotations folder. These subfolders at the root of this
    folder are:
	- ./raw - contains data as acquired from LabVIEW, without any video-IDs.
	- ./initial - holds mat files generated from raw data.
	- ./interpolated - interpolated data containing video-IDs.
	- ./non-interpolated - non-interpolated data containing video-IDs.

2. **/metadata**: this folder contains other information about the experiments.
    For example, information about the participants, the sequence in which they
    watched the videos, etc.

3. **/scripts**: contains scripts that allow the user to undertake/verify the 
    steps required for converting raw data to the processed data contained in
    the interpolated and non-interpolated folders.   


### 3. Usage:

**Note**:

- This repository only contains *raw* data and the accompanying code
that can be used to generate both the *interpolated* and *non-interpolated* data.

- The *raw* data doesn't contain any video label (video-IDs) information, hence
we do not recommend that users undertake downstream analysis with that data.
They should instead generate *interpolated* data,  and use that to perform any
downstream analysis.

- The *non-interpolated* data is can also be generated, in case researchers want
to use a different interpolation approach or prefer the non-interpolated data
for other reasons. 



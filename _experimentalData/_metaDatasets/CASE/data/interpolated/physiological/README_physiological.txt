%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            README -- CASE_dataset/data/interpolated/physiological
%
% This short guide to the interpolated data, covers the following topics:
% (1) General Information.
% (2) Extra Information.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------------------------------------------
(1) General Information:
-------------------------------------------------------------------------------
This folder contains the interpolated physiological data for all 30 subjects.
Each file (e.g., sub_1.csv) contains the following 10 comma-separated variables
(1 variable per column):

(1) daqtime: is the time provided by LabVIEW while logging. It is the global
    time and is also used for annotation files to allow synchronization of
    data across the two files. It is named daqtime to keep the variable name
    different from jstime (used in annotation data). Measurements in
    milliseconds (ms).
(2) ecg: Electrocardiogram sensor. Measurements in millivolts (mV).
(3) bvp: Blood Volume Pulse sensor. Measurements in BVP percentage.
(4) gsr: Galvanic Skin Response sensor. Measurements in microSiemens (uS).
(5) rsp: Respiration sensor. Measurements in respiration percentage.
(6) skt: Skin Temperature sensor. Measurements in degree Celsius (°C).
(7) emg_zygo: Surface Electromyography (sEMG) sensor placed on the Zygomaticus
              major muscles. Measurements in microvolts (uV).
(8) emg_coru: Surface Electromyography (sEMG) sensor placed on the Corrugator
              supercilli muscles. Measurements in microvolts (uV).
(9) emg_trap: Surface Electromyography (sEMG) sensor placed on the Trapezius
              muscles. Measurements in microvolts (uV).
(10) video: the video-IDs, that are repeated for the entire duration that a
            video appeared in the video-sequence for that participant.

-------------------------------------------------------------------------------
(2) Extra Information:
-------------------------------------------------------------------------------
The physiological sensors (i.e., number 2 to 9 in the list above) are plugged
into the sensor isolators (supplied by the sensor manufacturer) to ensure the
safety of the participants. The output from these sensor isolators is then
acquired through a DAQ card and logged using LabVIEW. The logged raw data stores
the acquired values in volts, that need to be converted into the desired
units for each sensor (see the list above). This is done using the formulas
provided by the manufacturer. Please see the data descriptor for more details
on this topic.

PLEASE NOTE: The physiological data in this folder is interpolated. Please see
the README file one folder up for more information on this topic.

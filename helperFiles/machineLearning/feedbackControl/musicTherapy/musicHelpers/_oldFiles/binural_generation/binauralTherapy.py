#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:42:46 2022

@author: samuelsolomon
"""

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import csv
import json
import numpy as np

# Audio Modules
from pygame import mixer
from scipy.io import wavfile
from scipy.io.wavfile import write

# Plotting Modules
import matplotlib.pyplot as plt

import sys

# -------------------------------------------------------------------------- #
# ----------------------------- Generate Sound ----------------------------- #

class generateSound:
    
    def __init__(self, musicFolder):
        self.noteFrequencies = self.getNoteFrequencies()
        self.musicFolder = musicFolder
        
        # Make an output folder for generated sounds.
        os.makedirs(self.musicFolder, exist_ok=True)

    def heuristicMap(self): # TODO
        self.map = {}
        return

    def getNoteFrequencies(self):   
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
        base_freq = 440 #Frequency of Note A4
        keys = np.array([x+str(y) for y in range(0,9) for x in octave])
        # Trim to standard 88 keys
        start = np.where(keys == 'A0')[0][0]
        end = np.where(keys == 'C8')[0][0]
        keys = keys[start:end+1]
        
        noteFrequencies = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
        noteFrequencies[''] = 0.0 # stop
        return noteFrequencies
    

    def getWaveform(self, frequencyLeft, frequencyRight, duration, samplingRate=44100, amplitude=4096):
        t = np.linspace(0, duration, int(samplingRate * duration), False)  # Time vector

        # Generate waveforms for each ear
        waveLeft = amplitude * np.sin(2 * np.pi * frequencyLeft * t)
        waveRight = amplitude * np.sin(2 * np.pi * frequencyRight * t)

        # Combine into a stereo waveform
        stereoWave = np.vstack((waveLeft, waveRight)).T
        return stereoWave

    def plotWave(self, waveform):
        # Plot sound wave
        plt.plot(waveform)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Sound Wave on Piano')
    

    def saveSound(self, waveform, filename, samplingRate=44100):
        # Adjusted to save stereo files
        write(self.musicFolder + filename, samplingRate, waveform.astype(np.int16))

        
    def loadSound(self, filename):
        # Load data from wav file
        samplingRate, waveform = wavfile.read(self.musicFolder + filename) 
        
        return samplingRate, waveform

                
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    plot = False

    # Initialize pygame mixer for stereo playback
    mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    
    # Specify the directory to save the generated sound
    generatedSoundsFolder = "./Organized Sounds/Generated Sounds/"
    
    # Create an instance of the generateSound class with the specified music folder
    generateSoundController = generateSound(musicFolder=generatedSoundsFolder)

    # Generate binaural beat waveform (No change in these lines)
    frequencyLeft = 72.5  # Example frequency for left ear
    delta = 6.3
    frequencyRight = frequencyLeft + delta  # Slightly different frequency for right ear
    duration = 60*4  # Duration in seconds

    print("Base Frequency = ", frequencyLeft)
    print("delta = ", delta)

    # Since we now need individual waveforms for plotting, let's generate them separately
    t = np.linspace(0, duration, int(44100 * duration), False)  # Time vector for the duration of the sound
    waveRight = 4096 * np.sin(2 * np.pi * frequencyRight * t)
    waveLeft = 4096 * np.sin(2 * np.pi * frequencyLeft * t)

    if plot:
        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(t, waveLeft, label=f'Left Ear Frequency: {frequencyLeft}Hz', color='blue')
        plt.plot(t, waveRight, label=f'Right Ear Frequency: {frequencyRight}Hz', color='red', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Binaural Beat Waveforms')
        plt.legend()
        plt.xlim(0, 0.05)  # Limit x-axis to show the waveform clearly
        plt.show()

    # Combine waveforms into a stereo waveform for saving and playback
    stereoWave = np.vstack((waveLeft, waveRight)).T

    # Generate binaural beat waveform
    binauralBeatWaveform = generateSoundController.getWaveform(frequencyLeft, frequencyRight, duration=duration)  # Duration in seconds
    

    # Save the binaural beat as a stereo sound file
    binauralBeatFilename = "binauralBeat.wav"
    generateSoundController.saveSound(binauralBeatWaveform, binauralBeatFilename)
    
    # Load and play the generated binaural beat sound
    soundFilePath = generatedSoundsFolder + binauralBeatFilename
    mixer.music.load(soundFilePath)
    mixer.music.play()

    # Keep the script running until the music is playing
    import time
    while mixer.music.get_busy():
        time.sleep(1)

    # Optionally, stop playback if needed
    mixer.music.stop()
    
    # Close the mixer
    mixer.quit()

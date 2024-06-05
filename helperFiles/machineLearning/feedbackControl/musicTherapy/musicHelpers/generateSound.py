""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os


class generateSound:

    def __init__(self, musicFolder):
        self.noteFrequencies = self.getNoteFrequencies()
        self.musicFolder = musicFolder

        # Make an output folder for generated sounds.
        os.makedirs(self.musicFolder, exist_ok=True)

    @staticmethod
    def getNoteFrequencies():
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
        base_freq = 440  # Frequency of Note A4
        keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
        # Trim to standard 88 keys
        start = np.where(keys == 'A0')[0][0]
        end = np.where(keys == 'C8')[0][0]
        keys = keys[start:end + 1]

        noteFrequencies = dict(zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))]))
        noteFrequencies[''] = 0.0  # stop
        return noteFrequencies

    def getWaveform(self, waveformFrequencyList, duration, waveformSamplingRate=44100, amplitude=4096):
        t = np.linspace(0, duration, int(waveformSamplingRate * duration))  # Time axis

        wave = np.zeros(len(t))
        for frequency in waveformFrequencyList:
            wave += amplitude * np.sin(2 * np.pi * self.noteFrequencies[frequency] * t)
        return wave

    @staticmethod
    def plotWave(waveformData):
        # Plot sound wave
        plt.plot(waveformData)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Sound Wave on Piano')

    def saveSound(self, waveformData, filename):
        wavfile.write(self.musicFolder + filename, rate=44100, data=waveformData.astype(np.int16))

    def loadSound(self, filename):
        # Load data from wav file
        waveformSamplingRate, waveformData = wavfile.read(self.musicFolder + filename)

        return waveformSamplingRate, waveformData


if __name__ == "__main__":
    # Initialize the class
    generatedSoundsFolder = "./Organized Sounds/Generated Sounds/"
    generateSoundController = generateSound(musicFolder=generatedSoundsFolder)  # Generates music to play

    # Make a new sound
    frequencyList = ['F3', 'A3']
    newSound = generateSoundController.getWaveform(frequencyList, duration=2, amplitude=2048)

    plt.plot(newSound);
    plt.xlim(0, 10000)
    plt.show()

    # Save and load the sound
    firstSoundFile = "firstSound.mp4"
    generateSoundController.saveSound(newSound, filename=firstSoundFile)
    samplingRate, waveform = generateSoundController.loadSound(firstSoundFile)
    # Play the new sounds.
    soundManager.loadSound(generatedSoundsFolder + firstSoundFile)
    soundManager.playSound()

    # Make a new sound
    frequencyList = ['B1', 'E4']
    newSound = generateSoundController.getWaveform(frequencyList, duration=2, amplitude=2048)

    plt.plot(newSound);
    plt.xlim(0, 10000)
    plt.show()

    # Save and load the sound
    secondSoundFile = "secondSound.mp4"
    generateSoundController.saveSound(newSound, filename=secondSoundFile)
    samplingRate, waveform = generateSoundController.loadSound(secondSoundFile)

""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import matplotlib.pyplot as plt
from pygame import mixer

from helperFiles.machineLearning.feedbackControl.musicTherapy.musicTherapy import generateSound


class soundController:

    def __init__(self):
        # Initialize the music player
        self.musicPlayer = mixer
        self.musicPlayer.init()

        self.songQueue = []

    def queueSounds(self, soundFiles):
        self.songQueue.extend(soundFiles)

    def playNextQueuedSong(self):
        self.loadSound(self.songQueue.pop(0))

    def loadSound(self, songFile):
        if songFile is not None:
            self.musicPlayer.music.load(songFile)  # Load the Sound into the player
            self.playSound()
        else:
            self.stopSound()

    def playNext(self, songFile):
        self.musicPlayer.music.queue(songFile)  # Queue the Sound

    def playSound(self):
        self.musicPlayer.music.play()  # Play the Sound

    def pauseSound(self):
        self.musicPlayer.music.pause()  # Pause the Sound

    def resumeSound(self):
        self.musicPlayer.music.unpause()  # Resume the Sound

    def stopSound(self):
        self.musicPlayer.music.stop()  # Stop the Sound

    def restartSound(self):
        self.musicPlayer.music.rewind()  # Restart the Sound

    def closeController(self):
        self.musicPlayer.quit()


if __name__ == "__main__":
    # Initialize the classes
    soundManager = soundController()  # Controls the music playing

    # Initialize the class
    generatedSoundsFolder = "./organizedSounds/generatedSounds/"
    generateSoundController = generateSound(musicFolder=generatedSoundsFolder)  # Generates music to play

    # Make a new sound
    frequencyList = ['72.5', '78.8']
    newSound = generateSoundController.getWaveform(frequencyList, duration=10, amplitude=2048)

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

    import numpy as np


    class BinauralBeatsGenerator:
        def __init__(self):
            self.noteFrequencies = {}  # Define your note frequencies here if needed

        def getWaveform(self, waveformFrequencyList, duration, waveformSamplingRate=44100, amplitude=4096):
            t = np.linspace(0, duration, int(waveformSamplingRate * duration), dtype=np.float32)  # Time axis
            wave = np.zeros(len(t), dtype=np.float32)
            for frequency in waveformFrequencyList:
                freq = self.noteFrequencies.get(frequency, frequency)
                wave += amplitude * np.sin(2.0 * np.pi * float(freq) * t)
            return wave


    # Example usage
    generator = BinauralBeatsGenerator()
    waveform = generator.getWaveform(['72.5', '78.8'], 100)

    # Save and load the sound
    firstSoundFile = "firstSound.mp4"
    generateSoundController.saveSound(waveform, filename=firstSoundFile)
    samplingRate, waveform = generateSoundController.loadSound(firstSoundFile)
    # Play the new sounds.
    soundManager.loadSound(generatedSoundsFolder + firstSoundFile)
    soundManager.playSound()



    #
    # # Make a new sound
    # frequencyList = ['B1', 'E4']
    # newSound = generateSoundController.getWaveform(frequencyList, duration=2, amplitude=2048)
    #
    # plt.plot(newSound);
    # plt.xlim(0, 10000)
    # plt.show()
    #
    # # Save and load the sound
    # secondSoundFile = "secondSound.mp4"
    # generateSoundController.saveSound(newSound, filename=secondSoundFile)
    # samplingRate, waveform = generateSoundController.loadSound(secondSoundFile)

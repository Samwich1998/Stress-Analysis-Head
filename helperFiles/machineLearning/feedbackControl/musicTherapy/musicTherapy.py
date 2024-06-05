""" Written by Samuel Solomon: https://scholar.google.com/citations?user=9oq12oMAAAAJ&hl=en """

import matplotlib.pyplot as plt
from scipy.io import wavfile
from pygame import mixer
import numpy as np
import json
import csv
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
        t = np.linspace(0, duration, int(waveformSamplingRate * duration), dtype=np.float32)  # Time axis

        wave = np.zeros(len(t), dtype=np.float32)
        for frequency in waveformFrequencyList:
            freq = self.noteFrequencies.get(frequency, frequency)
            wave += amplitude * np.sin(2.0 * np.pi * float(freq) * t)
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


# -------------------------  MTG-Jamendo Interface ------------------------- #

class compileMtgJamendo:

    def __init__(self, dataFolderMTG, soundInfoFileMTG):
        # General parameters
        self.dataFolder = dataFolderMTG  # The folder with the MTG-Jamendo dataset
        self.possibleTags = ['genre', 'instrument', 'mood/theme']  # The type of tags within the dataset

        # Organize songs by the tags
        self.genreMap = {}

        # Extract the sound information: locations and tags
        self.soundInfoLabels, fullSoundInfo = self.getSoundInfo(soundInfoFileMTG)
        self.soundInfo = self.cullSoundInfo(fullSoundInfo)

        self.organizeSounds(self.soundInfo)

    def getSoundInfo(self, soundInfoFile):
        # Open the MTG-Jamendo file with the metadata.        
        with open(self.dataFolder + soundInfoFile) as csvFile:
            csvReader = csv.reader(csvFile, delimiter="\t", quotechar='"')

            fullSoundInfo = []
            # For each track in the file
            for trackInfo in csvReader:
                # If it is the header 
                if 'TRACK_ID' in trackInfo:
                    # Save the labels
                    soundInfoLabels = trackInfo[0:5]
                    soundInfoLabels.append(self.possibleTags)
                    continue
                # Else, store the track information.
                fullSoundInfo.append(trackInfo[0:5])

                compiledTrackInfo = [[], [], []]
                # Organize the tags of the track by the tag type.
                for fullTag in trackInfo[5:]:
                    tagType, tag = fullTag.split("---")
                    index = self.possibleTags.index(tagType)
                    compiledTrackInfo[index].append(tag)
                fullSoundInfo[-1].append(compiledTrackInfo)
        return soundInfoLabels, fullSoundInfo

    def cullSoundInfo(self, fullSoundInfo):
        finalSoundInfo = []
        # For each track, store all the good ones.
        for trackInfo in fullSoundInfo:

            # Only take Songs longer than 4 minutes.
            if float(trackInfo[self.soundInfoLabels.index('DURATION')]) < 60 * 4:
                continue
            # Only take Songs less than 6 minutes.
            if float(trackInfo[self.soundInfoLabels.index('DURATION')]) > 60 * 6:
                continue
            # Only take songs with a genre tag
            if len(trackInfo[-1][self.possibleTags.index('genre')]) == 0:
                continue
            # Only take songs with a instrument tag
            if len(trackInfo[-1][self.possibleTags.index('instrument')]) == 0:
                continue
            # Only take songs with a mood/theme tag
            if len(trackInfo[-1][self.possibleTags.index('mood/theme')]) == 0:
                continue

            # Input the path to the sound
            trackInfo[3] = self.dataFolder + "audioFiles/" + os.path.basename(trackInfo[3])

            # Store the good tracks
            finalSoundInfo.append(trackInfo)

        return finalSoundInfo

    def organizeSounds(self, finalSoundInfo):
        self.genreMap = {}
        self.soundMap = {}
        # For each track, store all the good ones.
        for trackInfo in finalSoundInfo:
            trackNumber = trackInfo[0]

            # Hash the sounds by the genre.
            trackGenres = trackInfo[-1][0]
            for genre in trackGenres:
                genreTracks = self.genreMap.get(genre, [])
                genreTracks.append(trackNumber)

                self.genreMap[genre] = genreTracks

            # Hash sounds by its trackID
            self.soundMap[trackNumber] = trackInfo[1:]

    def plotTagDistribution(self, soundInfo):
        binNames = [];
        binHeights = []
        for trackInfo in soundInfo:

            # for tags in trackInfo[-1]:
            tags = trackInfo[-1][2]
            for tag in tags:
                if tag not in binNames:
                    binNames.append(tag)
                    binHeights.append(1)
                else:
                    binHeights[binNames.index(tag)] += 1

        print(binNames)
        plt.figure(figsize=(10, 5))
        # Create the bar plot
        plt.bar(binNames, binHeights, color='maroon', width=0.4)
        plt.xticks(rotation=90)  # Rotates X-Axis Ticks by 45-degrees

        plt.xlabel("Tagname")
        plt.ylabel("Number of occurences")
        plt.title("Tag Distribution in Sounds")
        plt.show()

    def plotFeatureTagRelation(self, soundInfo):
        tagList = [];
        feature1List = [];
        feature2List = []
        for trackInfo in soundInfo:

            # Get the audio tag.
            tags = sorted(trackInfo[-1][2])

            addTag = tags[-1]
            # if 'relaxing' in tags and not 'energetic' in tags:
            #     addTag = 'relaxing'
            # elif 'energetic' in tags and not 'relaxing' in tags:
            #     addTag = 'energetic'

            if addTag != None:
                tagList.append(addTag)
                # Get the audio features.
                featureFile = self.dataFolder + 'audioFeatures/' + os.path.basename(trackInfo[3]).split(".")[0] + ".json"
                featureData = self.readAcousticbrainzFeatures(featureFile)

                # print(featureData.keys())

                feature1List.append(featureData['rhythm']['onset_rate'])
                feature2List.append(featureData['tonal']['chords_changes_rate'])

        tagListUnique = list(np.unique(tagList))

        import collections
        classDistribution = collections.Counter(tagList)
        print(classDistribution)

        hashedTagList = []
        for tag in tagList:
            hashedTagList.append(tagListUnique.index(tag))

        plt.figure(figsize=(7, 5))
        # Create the bar plot
        plt.scatter(feature1List, feature2List, c=hashedTagList, s=6, cmap='bwr')
        # plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees

        plt.xlabel("onset_rate")
        plt.ylabel("chords_changes_rate")
        plt.title("Feature Seperation of the Tags")
        plt.show()

    def pullAudioInformation(self, soundInfo):
        pathToFolder = "../../../../../"
        for trackInfo in soundInfo:
            pathToTrack = pathToFolder + "audioFiles/" + os.path.basename(trackInfo[3]).split(".")[0] + ".mp3"
            pathToFeatures = pathToFolder + "audioFeatures/" + os.path.basename(trackInfo[3]).split(".")[0] + ".json"

            os.system("cp " + pathToTrack + " '" + os.path.dirname(trackInfo[3]) + "/'")
            os.system("cp " + pathToFeatures + " '" + "/".join(os.path.dirname(trackInfo[3]).split("/")[0:-1]) + "/audioFeatures/'")

    def readAcousticbrainzFeatures(self, featureFile):
        # Get the data from the json file
        with open(featureFile) as jsonFile:
            featureData = json.load(jsonFile)

        return featureData

    def pickSoundFromGenres(self, genres):
        print(self.genreMap.keys())
        soundFiles = []
        for genre in genres:
            if genre == None:
                soundFiles.append(None)
                continue

            # Randomly choose a song from the genre
            randomGenreHash = np.random.randint(0, high=len(self.genreMap[genre]))
            # Keep track of the random song from the genre
            trackID = self.genreMap[genre][randomGenreHash]
            soundFiles.append(self.soundMap[trackID][2])

        return soundFiles

    def getGenreList(self):
        return self.genreMap.keys()


# -------------------------------------------------------------------------- #
# ---------------------------- Sound Controller ---------------------------- #

class soundController(compileMtgJamendo):

    def __init__(self, dataFolder, soundInfoFile):
        super().__init__(dataFolder, soundInfoFile)

        # Initialize the music player
        self.musicPlayer = mixer
        self.musicPlayer.init()

        self.songQueue = []

    def queueSounds(self, soundFiles):
        self.songQueue.extend(soundFiles)

    def playNextQueuedSong(self):
        self.loadSound(self.songQueue.pop(0))

    def loadSound(self, songFile):
        if songFile != None:
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
    # Specify the MTG-Jamendo dataset path
    dataFolder = './Organized Sounds/MTG-Jamendo/'
    soundInfoFile = 'raw_30s_cleantags_50artists.tsv'
    # Initialize the classes
    soundManager = soundController(dataFolder, soundInfoFile)  # Controls the music playing

    # Extract the sound information
    soundInfo = soundManager.soundInfo
    # Collect the track information
    if False:
        soundManager.plotTagDistribution(soundInfo)
        soundManager.plotFeatureTagRelation(soundInfo)
    if False:
        soundManager.pullAudioInformation(soundInfo)

    # Play the music
    soundManager.loadSound(soundInfo[0][3])
    soundManager.playNext(soundInfo[1][3])
    # Play the music
    soundManager.playSound()
    # Pause and resume the music
    soundManager.pauseSound()
    soundManager.resumeSound()
    # Restart the music
    soundManager.restartSound()
    # Stop the music
    soundManager.stopSound()

    # # Close the controller
    # soundManager.closeController()

    if True:
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

    # import librosa
    # import librosa.display
    # musicData, samplingRate = librosa.load(soundInfo[2][3]) # Decodes audio into a 1-dimensional array, samplingRate of array

    # # Display the waveplot
    # librosa.display.waveshow(musicData, sr=samplingRate)

    # plt.figure(figsize=(14, 5))
    # # Display Spectrogram
    # musicDataFFT = librosa.stft(musicData)
    # musicDataFFT_DB = librosa.amplitude_to_db(abs(musicDataFFT))
    # librosa.display.specshow(musicDataFFT_DB, sr=samplingRate, x_axis='time', y_axis='hz') 
    # plt.colorbar()

    # zero_crossings = librosa.zero_crossings(musicData, pad=False)
    # numZeroCrossing = sum(zero_crossings)

# Heat Therapy hardware
# Therapy excel file saving
# Nucleate slides

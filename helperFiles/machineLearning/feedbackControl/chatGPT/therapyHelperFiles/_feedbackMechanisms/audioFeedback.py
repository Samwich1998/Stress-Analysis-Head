import sounddevice as sd
import speech_recognition as sr
import subprocess, threading
import numpy as np

class AudioFeedback:
    def __init__(self, client, threads, userName = "Sam"):
        self.client = client
        self.userName = userName
        # Initialize with settings or default values
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening
            # Adjust recognizer settings
        self.recognizer.pause_threshold = 0.3  # The length of silence (in seconds) at which it will consider that the speech has ended.
        self.recognizer.non_speaking_duration = 0.3  # The amount of silence (in seconds) that the recognizer allows before and after the actual speech. It helps in trimming the audio for processing.
        self.recognizer.energy_threshold = 4000  # The energy level (in an arbitrary unit) that the recognizer considers as speech versus silence/noise.
        self.recognizer.dynamic_energy_threshold = True  # Whether to adjust the energy_threshold automatically based on the ambient noise level.
        self.recognizer.dynamic_energy_adjustment_damping = 0.15 #  Controls how quickly the recognizer adjusts to changing noise conditions when dynamic_energy_threshold is True.
        self.recognizer.dynamic_energy_ratio = 1.5 # The factor by which speech is considered "louder" than the ambient noise when dynamic_energy_threshold is True.
        self.recognizer.operation_timeout = 10 # The maximum number of seconds the operation can run before timing out. None will wait indefinitely.
        self.recognizer.phrase_time_limit = 10 # The maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached.
        # Add any other initialization code here

        self.playAudioEvent, self.userRecordingEvent = threads
        self.noMoreSentencesBuffer = None

        self.setup_ffmpeg_process()

    def setup_ffmpeg_process(self):
        # Start a subprocess that uses ffmpeg to decode MP3 data to PCM and pipe the output to stdout for reading
        self.process = subprocess.Popen(['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', 'pipe:1'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        
    # ---------------------------------------------------------------------- #
    # ---------------------------- Audio Methods --------------------------- #
    
    def getAudioResponse(self, textPrompt, save_path = ""):        
        with self.client.audio.speech.with_streaming_response.create(
            response_format = 'mp3',  # The format to audio in. Supported formats are mp3, opus, aac, and flac.
            input = textPrompt,  # The text to generate audio for. The maximum length is 4096 characters.
            model = "tts-1",     # One of the available TTS models: tts-1 or tts-1-hd
            voice = "echo",     # The voice to use when generating the audio. Supported voices are alloy, echo, fable, onyx, nova, and shimmer.
            speed = 1,           # The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_bytes(chunk_size=2048):
                    yield chunk
                
                
    def audioCallback(self, outdata, frames, time, status):
        # Read decoded audio data from ffmpeg stdout and play it
        data = self.process.stdout.read(frames * 2)  # 2 bytes per frame for s16le format
        # print(len(data), len(outdata))
        if len(data) < 2 * len(outdata):
            outdata[:(len(data)//2)] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            # outdata[(len(data)//2):] = b'\x00' * (len(outdata) - len(data))  # Fill the rest with silence
            outdata[(len(data)//2):] = 0
            self.playAudioEvent.set()  # Signal that playback is finished
            raise sd.CallbackStop  # Stop the stream if we run out of data
        else:
            outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
        
        
    # Use a non-blocking approach to capture audio
    def voiceRecognitionCallback(self, recognizer, audio):
        print("I am ready to listen!")
        try:
            self.userPrompt = recognizer.recognize_google(audio)
            print("Here is what I heard: " + self.userPrompt)
            self.userRecordingEvent.set()  # Signal that we have recognized the audio
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            self.userRecordingEvent.set()  # Ensure the event is set on request error
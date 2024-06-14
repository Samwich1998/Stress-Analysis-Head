# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import subprocess, threading
import numpy as np
import os

from datetime import date
import random

import sounddevice as sd
import speech_recognition as sr

# OpenAI
from openai import OpenAI

# Import Files for Machine Learning
import browserControl      # Methods for controlling the web browser.
import imageModifications  # Methods for working with and altering images.

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #        

class chatGPTController:
    def __init__(self, userName = "Sam"):
        # General model parameters.
        self.imageModel = "dall-e-3"
        self.textEngine = "gpt-4-0613" # See text models at https://platform.openai.com/docs/models/gpt-4
        self.userName = userName        # A unique username for the client. Not used for personalization.
        
        # Set up the OpenAI API client
        self.client = OpenAI(api_key = "".join("sk-VGL24 JY EBqKOSb8P KF0ST3BlbkFJg BTO0uk UKc HUbaJ HlLel".split(" ")))
        
        self.noMoreSentencesBuffer = None
        
        
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
    
        self.userPrompt = None
        self.playback_finished = threading.Event()
        
        # Instantiate necesarry classes.
        self.browserController = browserControl.browserControl()
        
        # Create an Event object that will be set when audio is recognized
        self.userRecordingEvent = threading.Event()
        self.imageGenerationEvent = threading.Event()

        self.resetTherapySession()
        
        self.setup_ffmpeg_process()
        
    def setup_ffmpeg_process(self):
        # Start a subprocess that uses ffmpeg to decode MP3 data to PCM and pipe the output to stdout for reading
        self.process = subprocess.Popen(['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', 'pipe:1'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)



    def resetTherapySession(self):
        self.conversationHistory = [{
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a friendly and helpful virtual therapist. You empathize with what I am saying. In your first response to your patient, please introduce yourself and say your name is zenBot (only once). \
                             Be conversational as you talk with the user and build off of their ideas/words; however. \
                             Occasionally prompt the user for feedback on how you have helped them/how they are feeling. Always be sure to keep the conversation flowing. \
                             You will not directly hear the users verbal responses, only their state anxiety scores from the STAI-Y1 exam (range: 20 -80) \
                            Sometimes, you will be given the emotion profile and STAI score of a user. Don't explicitly say this information back to the user, but you can reference them getting more relaxed if your feedback is helping",
                 }],
            }]
             
    def changeUsers(self, userName):
        self.userName = userName
        
        self.resetTherapySession()

    def addEmotionInformation(self, STAI_score):
        
        self.conversationHistory.append(self.createChatHistory("system", 
                        f'Patient {self.userName} has a State-Trait Anxiety Inventory (STAI) score of {STAI_score}.'))
        
            
    # ---------------------------------------------------------------------- #
    # ---------------------------- Image Methods ---------------------------- #
        
    def imageThread(self, textPrompt, resultContainer):
        textPromptForImage = self.prepPromptForImage(textPrompt)
        response = self.getImageResponse(textPromptForImage)
        self.displayImage(response)
        self.saveImage(response, textPrompt)
        resultContainer['response'] = response
        return response

    def prepPromptForImage(self, textPrompt):
        i = -2
        mostRecentAssistantContent = []
        while len(mostRecentAssistantContent) < 2:
            if self.conversationHistory[i]['role'] != 'user':
                mostRecentAssistantContent.append(self.conversationHistory[i])
            i -= 1
        textPromptForImage = f"Previously, you said: {mostRecentAssistantContent[0]['content'][0]['text']} and {mostRecentAssistantContent[1]['content'][0]['text']}. Now, \
            the user said: {textPrompt}. As a friendly and helpful virtual therapist, generate a relevant \
            image to show the patient. You are trying to help them feel good."
        print(textPromptForImage)
        return textPromptForImage

    def getImageResponse(self, textPrompt):
        # Assert the proper data format.
        assert len(textPrompt) <= 4000, f"The maximum length is 4000 characters for text. Given {len(textPrompt)} characters"
        assert isinstance(textPrompt, str), f"Expecting the text prompt to be a string. Given type {type(textPrompt)}. Value: {textPrompt}"
        
        self.imageGenerationEvent.clear()
        
        # Interface with chatGPT API.
        response = self.client.images.generate(
            model = self.imageModel,
            response_format = "url",  # The format in which the generated images are returned. Must be one of url or b64_json.
            user = self.userName,     # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
            prompt=textPrompt,        # A text description of the desired image(s). The maximum length is 4000 characters.
            size="1024x1024",         # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            style="vivid",            # vivid: hyper-real and dramatic images; natural: natural, less hyper-real looking images
            quality="hd",
            n=1,                      # The number of images to generate. Must be between 1 and 10.
        )
        
        return response
    
    
    def displayImage(self, response):
        # Get the image URL.
        image_url = self.getImageURL(response)
        # Open the image URL with the webdriver.
        self.browserController.open_url(image_url)

    def saveImage(self, response, org_prompt, save_path="/_savedImages/"):
        # Get the image from image URL.
        image_url = self.getImageURL(response)
        imageRGBA = imageController.pullDownWebImage(image_url)

        # make image path
        filepath = os.path.dirname(__file__) + save_path
        if not os.path.exists(filepath):
            print('path does not exist', filepath)

        # save the file with todays date
        image_filepath = os.path.join(filepath, f"{date.today()}_{org_prompt}.png")
        imageRGBA.save(image_filepath, 'PNG')
        print(f"Image saved to {image_filepath}")

        return image_filepath
        
    
    def getImageURL(self, response):
        return response.channelData[0].url
    
    # ---------------------------------------------------------------------- #
    # ---------------------------- Text Methods ---------------------------- #
    
    def decodeTextResponse(self, textResponseObject):
        # If you have the full text ready.
        if hasattr(textResponseObject, 'choices'):
            return textResponseObject.choices[0].message.content
        else:
            fullText = ""
            for chunk in textResponseObject:
                fullText = fullText + chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content)
            
            return fullText
        
    def yieldFullSentence(self, textResponseObject):
        content_buffer = ""  # Initialize a buffer to hold the content
        for chunk in textResponseObject:
            content = chunk.choices[0].delta.content
            if content == None: continue
            
            if content not in ["!", ".", "\n"]:
                content_buffer += content
            else:
                content_buffer += content
                yield content_buffer
                content_buffer = ""
                
        yield self.noMoreSentencesBuffer
        
    def createChatHistory(self, speakerRole, textPrompt, imageResponse = None):
        currentChat = {
            "role": speakerRole, 
            "name": self.userName,
            "content": [
                {
                    "type": "text",
                    "text": textPrompt,
                },
            ],
        }
        
        if imageResponse != None:
            currentChat['content'].append({
                    "type": "image_url",
                    "image_url": {"url": self.getImageURL(imageResponse), "detail": "low"},
            })
        
        return currentChat
            
    def getTextReponse(self, textPrompt):        
        # Generate a response
        textResponseObject = self.client.chat.completions.create(
            messages=self.conversationHistory,      # A list of messages comprising the conversation so far.
            # response_format = "text",     # Must be one of text or json_object.
            model="gpt-4-vision-preview",            # ID of the model to use.
            frequency_penalty = 0,      # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            presence_penalty = 0,       # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
            user = self.userName,       # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
            temperature = 1,    # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
            stream = True,      # If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Example Python code.
            n = 1,              # How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
            max_tokens=300,
        )
        
        return textResponseObject
    
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
            self.playback_finished.set()  # Signal that playback is finished
            raise sd.CallbackStop  # Stop the stream if we run out of data
        else:
            outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)

    def playResponse(self, textResponseObject): 
        fullResponse = ""
        self.playback_finished.clear()
        # Open a sounddevice stream for playback
        with sd.OutputStream(callback=self.audioCallback, samplerate=44100, channels=1, dtype='int16'):
            while not textResponseObject.response.is_closed:
                textPrompt = self.yieldFullSentence(textResponseObject)
                
                textForAudio = next(textPrompt)
                if textForAudio == self.noMoreSentencesBuffer: 
                    continue
                print("\t", textForAudio)
                fullResponse += textForAudio
                    
                audio_chunks = self.getAudioResponse(textForAudio)
                
                # Feed MP3 data chunks to ffmpeg for decoding
                for chunk in audio_chunks:
                    self.process.stdin.write(chunk)
            
            # Close stdin to signal EOF to ffmpeg
            self.process.stdin.flush() # Helps clear the buffer.
            self.playback_finished.wait()
            # self.process.stdout.flush() # Will error out if no more inputs come.
            self.process.stdin.close() # Will error out if no more inputs come.
            self.process.wait()  # Needed to finish saying all the sentences.
        self.setup_ffmpeg_process()
        
        return fullResponse
        
        
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
                
    def getUserResponse(self):
        self.userPrompt = None
        self.userRecordingEvent.clear()  # Reset the event
        stop_listening = self.recognizer.listen_in_background(sr.Microphone(), self.voiceRecognitionCallback)
        
        self.userRecordingEvent.wait()  # Wait here until the event is set in the callback
        stop_listening(wait_for_stop=False)  # Stop listening
        
        # Add the user's response to the conversation history.
        self.conversationHistory.append(self.createChatHistory("user", self.userPrompt))
        
        return self.userPrompt

# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    gptController = chatGPTController()
    imageController = imageModifications.imageModifications(os.path.dirname(__file__) + "/_savedImages/")
    
    while True:
        # debug: generate random STAI score
        random_debug_score = random.randint(20, 80)
        print("STAI_Score =", random_debug_score)
        # Read in information about the user.
        gptController.addEmotionInformation(random_debug_score)
        #userPrompt = gptController.getUserResponse()
        userPrompt = ""
        gptController.conversationHistory.append(gptController.createChatHistory("user", userPrompt))
                
        # Ask chatGPT to generate a text and image response.
        imageResponse = {}
        thread = threading.Thread(target=gptController.imageThread, args=(userPrompt,imageResponse))
        thread.start()
        # imageResponseObject = gptController.getImageResponse(userPrompt) 
        textResponseObject = gptController.getTextReponse(userPrompt) 
                                
        # Read out the response to the user in real-time.
        fullAudioResponse = gptController.playResponse(textResponseObject)
        # gptController.displayImage(imageResponseObject)
        # threading.Thread(target=gptController.playResponse, args=(imageResponseObject,)).start()
        # threading.Thread(target=gptController.displayImage, args=(textResponseObject,)).start()
        thread.join()
        # Keep track of the full conversation.
        gptController.conversationHistory.append(gptController.createChatHistory("assistant", fullAudioResponse, imageResponse.get('response')))
        # print(gptController.conversationHistory)

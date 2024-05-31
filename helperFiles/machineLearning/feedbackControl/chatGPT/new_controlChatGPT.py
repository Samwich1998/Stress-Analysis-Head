# General
import speech_recognition as sr
from datetime import date
import sounddevice as sd
from openai import OpenAI
import threading
import textwrap
import json
import os

# Import helper classes
from therapyHelperFiles._feedbackMechanisms.audioFeedback import AudioFeedback
from therapyHelperFiles._feedbackMechanisms.imageFeedback import ImageFeedback
from therapyHelperFiles._feedbackMechanisms.textFeedback import TextFeedback
from therapyHelperFiles.simulatedEmotion import simulatedEmotion


class chatGPTController:
    def __init__(self, userName):
        self.conversationHistory = None
        self.getMicPrompt = False

        # General model parameters.
        self.demographics = simulatedEmotion.generate_random_demographics()
        self.textEngine = "gpt-4-0613"  # See text models at https://platform.openai.com/docs/models/gpt-4
        self.imageModel = "dall-e-3"  # See image models at https://platform.openai.com/docs/models/dall-e-3
        self.userName = userName  # A unique username for the client. Not used for personalization.

        # Set up the OpenAI API client
        self.client = OpenAI(api_key="".join("sk-VGL24 JY EBqKOSb8P KF0ST3BlbkFJg BTO0uk UKc HUbaJ HlLel".split(" ")))

        self.saveConversationFilePath = self.getsaveFilePath()
        self.userPrompt = None

        # Instantiate the necessary classes.
        self.imageGenerationEvent = threading.Event()
        self.userRecordingEvent = threading.Event()
        self.playAudioEvent = threading.Event()

        self.audioFeedback = AudioFeedback(self.client,
                                           threads=[self.playAudioEvent, self.userRecordingEvent],
                                           userName=self.userName)
        self.imageFeedback = ImageFeedback(self.client,
                                           self.imageModel,
                                           self.imageGenerationEvent,
                                           self.saveConversationFilePath,
                                           userName=self.userName)
        self.textFeedback = TextFeedback(self.client,
                                         self.textEngine,
                                         userName=self.userName)

        self.resetTherapySession()

    @staticmethod
    def getsaveFilePath(save_path="/Helper Files/_savedConversations/"):
        filepath = os.path.dirname(__file__) + save_path
        saveFilePath = os.path.join(filepath, f"{date.today()}")
        counter = 0
        while os.path.exists(f'{saveFilePath}_{counter}.txt'):
            counter += 1

        return f'{saveFilePath}_{counter}.txt'

    def resetTherapySession(self, conversationFile, includeUserDemographic=True):
        text = "You are a friendly and helpful virtual therapist performing Cognitive Behavioral Therapy. In your first response, introduce yourself and say your name is zenBot (only once). \
                                You are having a one sided therapeutic conversation as you help a user to understand, control, and improve their mental state. \
                                You will not directly hear the users verbal responses, only their state anxiety scores from the STAI-Y1 exam (range: 20 - 80). Do not mention this score to the user. \
                                You will also be given the STAI and PANAS emotion profile. Do not mention these emotions back to the user."

        if includeUserDemographic:
            text += f" You are helping a patient of the following demographics: {self.demographics}."
        self.conversationHistory = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": text,
            }],
        }]

        conversationFile = open(self.saveConversationFilePath, 'w')
        conversationFile.write(json.dumps(self.conversationHistory[0], indent=4) + ", \n")
        conversationFile.close()

    def changeUsers(self, userName):
        self.userName = userName

        self.resetTherapySession()

    def addEmotionInformation(self, STAI_score, emotion_profile):

        self.conversationHistory.append(self.textFeedback.createChatHistory("system",
                                                                            f'Patient has a State-Trait Anxiety Inventory (STAI) score of {STAI_score}. \
                                Patient Emotion Profile from the STAI survey questions are {emotion_profile[0]} and \
                                from the PANAS survey questions are {emotion_profile[1]}. Do not mention the results of the emotion profile; \
                                instead provide a holistic review as you help a user to understand, control, and improve their mental state. \
                                Be extremely concise in your response, no more than 30 seconds.'))

    # ---------------------------- Text Methods ---------------------------- #

    def playResponse(self, textResponseObject):
        fullResponse = ""
        self.playAudioEvent.clear()
        # Open a sounddevice stream for playback
        with sd.OutputStream(callback=self.audioFeedback.audioCallback, samplerate=44100, channels=1, dtype='int16'):
            while not textResponseObject.response.is_closed:
                textPrompt = self.textFeedback.yieldFullSentence(textResponseObject)

                textForAudio = next(textPrompt)
                if textForAudio == self.textFeedback.noMoreSentencesBuffer:
                    continue
                print("\t", textForAudio)
                fullResponse += textForAudio

                audio_chunks = self.audioFeedback.getAudioResponse(textForAudio)

                # Feed MP3 data chunks to ffmpeg for decoding
                for chunk in audio_chunks:
                    self.audioFeedback.process.stdin.write(chunk)

            # Close stdin to signal EOF to ffmpeg
            self.audioFeedback.process.stdin.flush()  # Helps clear the buffer.
            self.playAudioEvent.wait()
            # self.process.stdout.flush() # Will error out if no more inputs come.
            self.audioFeedback.process.stdin.close()  # Will error out if no more inputs come.
            self.audioFeedback.process.wait()  # Needed to finish saying all the sentences.
        self.audioFeedback.setup_ffmpeg_process()

        return fullResponse

    def getUserResponse(self):
        self.audioFeedback.userPrompt = None
        self.userRecordingEvent.clear()  # Reset the event
        stop_listening = self.audioFeedback.recognizer.listen_in_background(sr.Microphone(), self.audioFeedback.voiceRecognitionCallback)

        self.userRecordingEvent.wait()  # Wait here until the event is set in the callback
        stop_listening(wait_for_stop=False)  # Stop listening
        # Add the user's response to the conversation history.
        self.userPrompt = self.audioFeedback.userPrompt
        self.conversationHistory.append(self.textFeedback.createChatHistory("user", self.userPrompt))
        print(self.userPrompt)

        return self.userPrompt


# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    gptController = chatGPTController(userName="Squirtle")
    while True:
        # to be implemented: get STAI score and emotion profile
        stai_score, currentEmotionProfile = simulatedEmotion.generate_random_emotion_profile()

        print("STAI_Score =", stai_score)
        print("Emotion Profile =", currentEmotionProfile)
        # Read in information about the user.
        gptController.addEmotionInformation(stai_score, currentEmotionProfile)
        if gptController.getMicPrompt:
            userPrompt = gptController.getUserResponse()
        else:
            userPrompt = ""
            gptController.conversationHistory.append(gptController.textFeedback.createChatHistory("user", userPrompt))
        # Ask chatGPT to generate a text and image response.
        imageResponse = {}
        thread = threading.Thread(target=gptController.imageFeedback.imageThread,
                                  args=(gptController.conversationHistory, imageResponse))
        thread.start()
        # imageResponseObject = gptController.getImageResponse(userPrompt) 
        outputTextResponseObject = gptController.textFeedback.getTextReponse(gptController.conversationHistory)

        # Read out the response to the user in real-time.
        fullAudioResponse = gptController.playResponse(outputTextResponseObject)
        thread.join()
        # Keep track of the full conversation.
        most_recent_chat = gptController.textFeedback.createChatHistory("assistant", fullAudioResponse, imageResponse.get('response'))
        gptController.conversationHistory.append(most_recent_chat)

        conversationFile = open(gptController.saveConversationFilePath, 'a')
        my_wrap = textwrap.TextWrapper(width=200)
        conversationFile.write(my_wrap.fill(json.dumps(most_recent_chat, indent=4)) + ", \n")
        conversationFile.close()

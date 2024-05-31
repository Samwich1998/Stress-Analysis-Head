# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import time

import speech_recognition as sr
import soundfile as sf
import sounddevice as sd


# OpenAI
from openai import OpenAI

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/Helper Files/")
import browserControl      # Methods for controlling the web browser.

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #

class gptAssistant:
    def __init__(self, userName = ""):
        # General model parameters.
        self.imageModel = "dall-e-3"
        self.textEngine = "gpt-4-0613" # See text models at https://platform.openai.com/docs/models/gpt-4
        self.userName =userName        # A unique username for the client. Not used for personalization.
        
        # Set up the OpenAI API client
        # OpenAI.api_key = "".join()
        self.client = OpenAI(api_key = "".join("sk-VGL24 JY EBqKOSb8P KF0ST3BlbkFJg BTO0uk UKc HUbaJ HlLel".split(" ")))
        
        # Instantiate necesarry classes.
        # self.browserController = browserControl.browserControl()
        
        self.createTherapyAssistant()
        self.beginUserConversation()
    
    def createTherapyAssistant(self):
        self.assistant = self.client.beta.assistants.create(
            name="ZenBot",
            description="Tele-mental health therapist.",
            instructions="You are a personal anxiety therapist. Have a conversation to the user and really listen to how they are feeling. Please help them relax.",
            tools=[{"type": "retrieval"}],
            model="gpt-4-vision-preview",
        )
        
    def beginUserConversation(self):
        self.clientThread = self.client.beta.threads.create()
        
    def askTherapyAssistant(self, userPrompt):
        message = self.client.beta.threads.messages.create(
            thread_id=self.clientThread.id,
            role="user",
            content=userPrompt,
        )
        
        return message
        
    def runTherapyAsistant(self):
        responseQuery = self.client.beta.threads.runs.create(
            thread_id=self.clientThread.id,
            assistant_id=self.assistant.id,
            instructions="Please address the user as Sam the Amazing. The user has a premium account."
        )
        
        return responseQuery
    
    def checkRunStatus(self, responseQuery):
        responseStatus = self.client.beta.threads.runs.retrieve(
            thread_id=self.clientThread.id,
            run_id=responseQuery.id
        )
        
        return responseStatus
    
    def getTherapistResponse(self):
        # Submit the question to the chatGPT server.
        responseQuery = self.runTherapyAsistant()
        
        # Periodically check for the response to come back.
        while self.checkRunStatus(responseQuery).completed_at == None:
            time.sleep(1)
        
        
        messages = self.client.beta.threads.messages.list(
            thread_id=self.clientThread.id,
        )
        
        return messages


    def getUserResponse(self):
        # Setup audio collection.
        recognizer = sr.Recognizer()
        userPrompt = None # Initialize an empty prompt
        
        # While the prompt is empty
        while userPrompt == None:
            # Open the microphone to listen for prompt
            with sr.Microphone() as source:
                print("Let me know how you are feeling!")
                audio = recognizer.listen(source)

            try:
                userPrompt = recognizer.recognize_google(audio)
                print("Here is what I heard: " + userPrompt)
            except:
                print("Nothing heard from the user. Try again.")
        
        return userPrompt
    
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    therapyAssistant = gptAssistant()
    
    prompts = [
        "I have a state anxiety score of 60 out of 80, a postive affectivity score of 10 out of 25, and a negative affectivity score of 18. Help me relax through a text response.",
    ]
            
    # ---------------------------------------------------------------------- #
    # ------------------------ Edit a Prompted Image ----------------------- #
    
    # userPrompt = therapyAssistant.getUserResponse()
    userPrompt = prompts[0]
    
    t1 = time.time()
    
    therpistResponse = therapyAssistant.askTherapyAssistant(userPrompt)
    
    response = therapyAssistant.getTherapistResponse()
    textResponse = response.data[0].content[0].text.value
    
    t2 = time.time()
    
    print(response, t2-t1)






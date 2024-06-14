# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import io
from datetime import date

import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
from playsound import playsound
import threading
from queue import Queue

from IPython.display import Audio


# OpenAI
from openai import OpenAI

# Import Files for Machine Learning
sys.path.append(os.path.dirname(__file__) + "/Helper Files/")
import browserControl      # Methods for controlling the web browser.
import imageModifications  # Methods for working with and altering images.

sys.path.append(os.path.dirname(__file__) + "/Helper Files/_feedbackMechanisms")
import audioFeedback
import imageFeedback
import textFeedback

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #

class chatGPTController:
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

        self.audioQueue = Queue()
        self.audioThread = threading.Thread(target=self.processAudioQueue)
        self.audioThread.start()
    
    # ---------------------------------------------------------------------- #
    # -------------------------- General Methods --------------------------- #
    
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

    
    def printModels(self):
        print(openai.Model.list())
        
    def getImageURL(self, response):
        return response.channelData[0].url
        
    # ---------------------------------------------------------------------- #
    # ---------------------------- Text Methods ---------------------------- #
        
    def getTextReponse(self, textPrompt, speak=True):
        # Generate a response
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": textPrompt}],
            stream=True,
        )
        content_buffer = ""  # Initialize a buffer to hold the content
        batching_symbols = '.:?!-'
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content != None and content != "":
                print(content, end="", flush=True)
                content_buffer += content
                if content_buffer[-1] in batching_symbols:
                    if speak:
                        self.audioQueue.put(content_buffer)
                        content_buffer = ""  # Clear the buffer after processing
        '''
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                if len(chunk.choices[0].delta.content) > 0:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    if speak:
                        audio_thread = threading.Thread(target=self.getAudioResponse, args=(chunk.choices[0].delta.content,))
                        audio_thread.start()
                        # self.getAudioResponse(chunk.choices[0].delta.content)
        '''
        return response
    
    def printTextString(self, response):
        text_response = response.choices[0].message.content
        # text_response = response['choices'][0]['message']['content']
        print(text_response)
        return text_response
    


    # ---------------------------------------------------------------------- #
    # ---------------------------- Audio Methods --------------------------- #

    def processAudioQueue(self):
        while True:
            # Wait for and get the next batch from the queue
            content_batch = self.audioQueue.get()
            if content_batch is None:
                break  # None is used as a signal to stop the thread
            self.getAudioResponse(content_batch)
            self.audioQueue.task_done()
    
    def getAudioResponse(self, textPrompt, save_path = "/_savedAudio/"):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=textPrompt,
        )
        filepath = os.path.dirname(__file__) + save_path
        audio_filepath = os.path.join(filepath, f"{date.today()}_output.mp3")
        print(audio_filepath)

        response.stream_to_file.method(audio_filepath)
        # After the file is saved, play it
        Audio(audio_filepath)

        '''
        buffer = io.BytesIO()
        for chunk in response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)

        with sf.SoundFile(buffer, 'r') as sound_file:
            data = sound_file.read(dtype='int16')
            sd.play(data, sound_file.samplerate)
            sd.wait()
        '''
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Image Methods ---------------------------- #
    
    def getImageResponse(self, textPrompt):
        # Assert the proper data format.
        assert len(textPrompt) <= 1000, f"The maximum length is 1000 characters for text. Given {len(textPrompt)} characters"
        assert isinstance(textPrompt, str), f"Expecting the text prompt to be a string. Given type {type(textPrompt)}. Value: {textPrompt}"
        
        # Interface with chatGPT API.
        response = self.client.images.generate(
            model = self.imageModel,
            response_format = "url",  # The format in which the generated images are returned. Must be one of url or b64_json.
            user = self.userName,     # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
            prompt=textPrompt,        # A text description of the desired image(s). The maximum length is 1000 characters.
            size="1024x1024",         # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            style="vivid",            # vivid: hyper-real and dramatic images; natural: natural, less hyper-real looking images
            quality="hd",
            n=1,                      # The number of images to generate. Must be between 1 and 10.
        )
        
        return response
    
    def varyImageResponse(self, image, mask, textPrompt):
        response = self.client.images.edit(
            response_format = "url",  # The format in which the generated images are returned. Must be one of url or b64_json.
            user = self.userName,     # A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
            prompt=textPrompt,        # A text description of the desired image(s). The maximum length is 1000 characters.
            size="1024x1024",         # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            image=image,              # The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.
            mask=mask,                # An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
            n=1,                      # The number of images to generate. Must be between 1 and 10.
        )
        
        return response
    
    # ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    gptController = chatGPTController()
    imageController = imageModifications.imageModifications(os.path.dirname(__file__) + "/_savedImages/")
    
    prompts = [
        "I have a state anxiety score of 60 out of 80, a postive affectivity score of 10 out of 25, and a negative affectivity score of 18. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
        "After your last image, my STAI state anxiety (20 - 80) score went from 60 to 80 out of 80, my postive affectivity score went from 10 to 15 out of 25, and my negative affectivity score went from 18 to 20 out of 25. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
        "After your last image, my STAI state anxiety (20 - 80) score went from 80 to 50 out of 80, my postive affectivity score went from 15 to 14 out of 25, and my negative affectivity score went from 20 to 15 out of 25. Generate an image that will reduce my anxiety GIVEN the anxiety scores I have told you. For example, you can display a lovely mountain range that is peaceful and transquil, use your judgement.",
    ]
    
    prompts = [
        # "Generate a calming image of a realistic beautiful beach.",
        # "Display a calming image of a realistic outdoor view of a snowy oasis on christmas night.",
        "Display a calming image of a realistic warm enviroment.",
        "This image made me less anxious. Please make me another image.",
        "This image made me more anxious. Please help me get out of this mindset.",
        "This image made me less anxious. Please make me another image.",
        # "Display a calming image of a realistic indoor view of a japenese zen house with a firepit, a koi pond, and the jungle.",
    ]
    
    # Flags for which programs to run.
    displayPromptedImages = False
    editPromptedImage = True
    testingPrompt = 'I am stressed. Help me feel better'
    
    # ---------------------------------------------------------------------- #
    # --------------------- Generate Images for Display -------------------- #
    
    if displayPromptedImages:
        # For each prompt.
        for prompt in prompts:
            # Ask chatGPT to generate an image response.
            response = gptController.getImageResponse(prompt)
            gptController.displayImage(response)
            
    # ---------------------------------------------------------------------- #
    # ------------------------ Edit a Prompted Image ----------------------- #
    
    if editPromptedImage:     
        
        while True:
            if testingPrompt:
                prompts = [testingPrompt]
            else:
                prompts = []
                while len(prompts) == 0:
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        print("Listening...")
                        audio = recognizer.listen(source)

                    try:
                        prompts = [recognizer.recognize_google(audio)]
                        print("Transcribed Text: " + prompts[0])
                    except Exception as e:
                        print("Error in speech recognition: ", e, "\nTrying again.")

            textResponse = gptController.getTextReponse(prompts[0])
            # textResponse_str = gptController.printTextString(textResponse)

            gptController.getAudioResponse(textResponse_str)
                
            # Ask chatGPT to generate an image response.
            initialResponse = gptController.getImageResponse(prompts[0])
            gptController.displayImage(initialResponse)
            gptController.saveImage(initialResponse, prompts[0])

            sys.exit()
        
        # Ask chatGPT to generate an image response.
        # initialResponse = gptController.getImageResponse(prompts[1])
        # gptController.displayImage(initialResponse)
        
        # # Ask chatGPT to generate an image response.
        # initialResponse = gptController.getImageResponse(prompts[2])
        # gptController.displayImage(initialResponse)
        
        # # Ask chatGPT to generate an image response.
        # initialResponse = gptController.getImageResponse(prompts[3])
        # gptController.displayImage(initialResponse)
        
        sys.exit()
        
        # Get the image content from the URL.
        image_url = gptController.getImageURL(initialResponse)
        imageRGBA = imageController.pullDownWebImage(image_url) # Convert the the chatGPT image format.
        
        # Make a mask for the image.
        imageMaskRGBA = imageController.make_top_half_translucent(imageRGBA)
        # imageMaskRGBA = imageController.remove_hex_color(imageRGBA, "#FFFFFF")
        # imageMaskRGBA = imageController.remove_similar_colors(imageRGBA, "#FFFFFF", tolerance = 250)
        
        # Conver the images into the correct chatGPT format.
        imageMask = imageController.rbga2ByteArray(imageMaskRGBA)
        imageByteArray = imageController.rbga2ByteArray(imageRGBA)

        # Regenerate the image with the mask filled in.
        # finalResponse = gptController.varyImageResponse(imageByteArray, imageMask, prompts[0])
        # gptController.displayImage(finalResponse)







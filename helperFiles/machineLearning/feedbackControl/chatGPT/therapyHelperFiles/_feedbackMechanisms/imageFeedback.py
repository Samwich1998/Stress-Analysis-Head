import os
from datetime import date

# Import Files for Machine Learning
import browserControl      # Methods for controlling the web browser.
import imageModifications  # Methods for working with and altering images.
import sys, os, json

class ImageFeedback:
    def __init__(self, client, model, thread, txtFilePath, userName = "Sam"):
        self.client = client
        self.imageModel = model
        self.imageGenerationEvent = thread
        self.userName = userName
        self.txtFilePath = txtFilePath

        self.browserController = browserControl.browserControl()
        self.imageController = imageModifications.imageModifications(os.path.dirname(__file__) + "/_savedImages/")


    # ---------------------------------------------------------------------- #
    # ---------------------------- Image Methods ---------------------------- #
        
    def imageThread(self, conversationHistory, resultContainer):
        textPrompt, textPromptForImage = self.prepPromptForImage(conversationHistory)
        response = self.getImageResponse(textPromptForImage)
        self.displayImage(response)
        if textPrompt == '':
            self.saveImage(response, textPromptForImage)
        else:
            self.saveImage(response, textPrompt)
        resultContainer['response'] = response
        return response

    def prepPromptForImage(self, conversationHistory):
        textPrompt = conversationHistory[-1]['content'][0]['text']
        i = -2
        mostRecentAssistantContent = []
        while len(mostRecentAssistantContent) < 2:
            if conversationHistory[i]['role'] != 'user':
                mostRecentAssistantContent.append(conversationHistory[i])
            i -= 1
        if textPrompt != '':
            textPromptForImage = f"Previously, you said: {mostRecentAssistantContent[0]['content'][0]['text']} and {mostRecentAssistantContent[1]['content'][0]['text']}. Now, \
                the user said: {textPrompt}. As a friendly and helpful virtual therapist, generate a relevant \
                image to show the patient. You are trying to help them feel good."
            return textPrompt, textPromptForImage
        else:
            textPromptForImage = f"Previously, you said: {mostRecentAssistantContent[0]['content'][0]['text']} and {mostRecentAssistantContent[1]['content'][0]['text']}. \
                As a friendly and helpful virtual therapist, generate a relevant image to show the patient. You are trying to help them feel good."
            return mostRecentAssistantContent[1]['content'][0]['text'][:20], textPromptForImage

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

    def saveImage(self, response, org_prompt, save_path="/Helper Files/_savedImages/"):
        # Get the image from image URL.
        image_url = self.getImageURL(response)
        imageRGBA = self.imageController.pullDownWebImage(image_url)

        # make image path
        filepath = os.path.dirname(__file__) + "/../../" + save_path
        if not os.path.exists(filepath):
            print('path does not exist', filepath)

        # save the file with todays date
        image_filepath = os.path.join(filepath, f"{date.today()}_{org_prompt}.png")
        imageRGBA.save(image_filepath, 'PNG')
        print(f"Image saved to {image_filepath}")

        # write this to conversation history txt file
        file = open(self.txtFilePath, 'a')
        print(self.txtFilePath)
        file.write(f"*** Image saved to {image_filepath} *** \n")
        file.close()

        return image_filepath
        
    def getImageURL(self, response):
        return response.data[0].url

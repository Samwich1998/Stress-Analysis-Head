class TextFeedback:
    def __init__(self, client, model, userName = "Sam"):
        self.client = client
        self.textEngine = model
        self.noMoreSentencesBuffer = None
        self.userName = userName

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
                    "image_url": {"url": imageResponse.data[0].url, "detail": "low"},
            })
        
        return currentChat
            
    def getTextReponse(self, conversationHistory):  
        print(conversationHistory)      
        # Generate a response
        textResponseObject = self.client.chat.completions.create(
            messages=conversationHistory,      # A list of messages comprising the conversation so far.
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
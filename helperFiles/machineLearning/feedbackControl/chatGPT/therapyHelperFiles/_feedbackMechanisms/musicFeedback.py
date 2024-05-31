

import speech_recognition as sr
import openai

# Function to listen and transcribe speech to text
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("Transcribed Text: " + text)
        return text
    except Exception as e:
        print("Error in speech recognition: ", e)
        return None

# Function to send text to ChatGPT and get a response
def get_chatgpt_response(text):
    openai.api_key = "".join("sk-VGL24 JY EBqKOSb8P KF0ST3BlbkFJg BTO0uk UKc HUbaJ HlLel".split(" "))
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": text}]
        )
    except Exception as e:
        print("Error in ChatGPT response: ", e)
        return None

# Main function to integrate speech recognition with ChatGPT
def main():
    user_input = listen_and_transcribe()
    if user_input:
        response = get_chatgpt_response(user_input)
        print("ChatGPT Response:", response)

if __name__ == "__main__":
    main()

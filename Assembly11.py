import assemblyai as aai
from openai import OpenAI
import time
import os
import pygame
import threading
import atexit
import asyncio
import edge_tts

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = "ceeca40708804bcfa78588a8b7b3349a"
        self.openai_client = OpenAI(api_key="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")
        self.transcriber = None
        self.full_transcript = []
        atexit.register(self.cleanup)
        
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )
        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)
    
    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Please speak now...")
        return
    
    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
    
    def on_error(self, error: aai.RealtimeError):
        print("An error occurred:", error)
        return
    
    def on_close(self):
        return
    
    def generate_ai_response(self, transcript):
        self.stop_transcription()
        user_text = transcript.text
        print(f"\nUser: {user_text}\n")
        self.full_transcript.append({"role": "user", "content": user_text})
        system_prompt = {
            "role": "system",
            "content": "You are a computer science teacher. Help students understand programming, algorithms, and theory. Answer questions in no more than 3-4 lines for clarity and conciseness."
                       "Your response will be converted to audio for the user. Please provide a response to the user's question and make sure if there are any mathematical expressions, they are read out in way that is easy to understand."
                       "Without any additional text, please provide a response to the user's question."
        }
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_prompt] + self.full_transcript,
            max_tokens=10000
        )
        
        ai_response = response.choices[0].message.content
        self.full_transcript.append({"role": "assistant", "content": ai_response})
        asyncio.run(self.generate_audio(ai_response))

    async def generate_audio(self, text):
        print(f"\nAI Teacher: {text}")
        communicate = edge_tts.Communicate(text, "en-AU-NatashaNeural")
        await communicate.save("response.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("response.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("response.mp3")
        self.prompt_user()

    def prompt_user(self):
        try:
            user_input = input("\nEnter 1 to ask a question or 0 to finish the conversation: ")
            if user_input == "1":
                self.start_transcription()
            elif user_input == "0":
                print("\nConversation Transcript:")
                for entry in self.full_transcript:
                    print(f"{entry['role'].capitalize()}: {entry['content']}")
                print("\nThank you for the conversation!")
                self.cleanup()
                os._exit(1)
            else:
                print("Invalid input. Please enter 1 or 0.")
                self.prompt_user()
        except KeyboardInterrupt:
            self.cleanup()
            print("\nProgram interrupted. Exiting.")
            os._exit(1)

    def cleanup(self):
        self.stop_transcription() 
        print("Cleaned up the system.")
        os._exit(1)
        
greeting = "Hey! How are you doing, I am your Computer Science teacher. How can I help you?"
ai_assistant = AI_Assistant()
asyncio.run(ai_assistant.generate_audio(greeting))

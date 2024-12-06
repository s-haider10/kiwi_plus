import assemblyai as aai
from openai import OpenAI
import time
import os
import pygame
import threading
import atexit
import asyncio
import edge_tts
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from yt_dlp import YoutubeDL
import os
import argparse
import subprocess
import cv2
from scenedetect import detect, ContentDetector
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
import chromadb
from chromadb.config import Settings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import pytesseract
import re
from datetime import timedelta
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import assemblyai as aai
from pydub import AudioSegment
from openai import OpenAI
import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sentence_transformers
from tqdm.autonotebook import tqdm, trange

class AI_Assistant:
    def __init__(self):
        # add the os.env
        aai.settings.api_key = "ceeca40708804bcfa78588a8b7b3349a"
        self.openai_client = OpenAI(api_key="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")
        self.transcriber = None
        self.full_transcript = []
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        self.stop_words = set(stopwords.words("english"))
        self.client_capstone = chromadb.PersistentClient(path="chromadb")
        self.collection_transcriptions = self.client_capstone.get_collection(name="Capstone_Kiwi")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.initialize_system_prompt()
        atexit.register(self.cleanup)
    
    def embed_text(self, text):
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**text_inputs)
        return text_embeddings.squeeze().cpu().numpy()
    
    def get_relevant_context(self, query_text):
        merged_info = ""
        query_embedding = self.embed_text(query_text)
        results = self.collection_transcriptions.query(query_embeddings=query_embedding, n_results=3)
        data_string = results['documents'][0]
        for i, item in enumerate(data_string):
            merged_info += f"Context {i+1}: \n\n"
            merged_info += f"\n\n{item}\n\n"
        return merged_info      
    
    def initialize_system_prompt(self):
        system_prompt = {
            "role": "system",
            "content": (
                "You are an AI teacher designed to assist students in understanding video content. "
                "When a student asks a question, analyze the provided context from the video transcription carefully "
                "and respond in a clear, engaging, and student-friendly manner. Match the language style and tone of the video "
                "to maintain a cohesive learning experience. Your answers should be concise if the query requires a direct response, "
                "limited to 2-3 sentences. For more complex or detailed questions, provide thorough explanations in 5-6 sentences, "
                "incorporating examples, quotes, or key points from the video transcription where relevant. "
                "Always aim to cite the exact timestamps (start and end) from the video to guide students to specific sections for additional clarity. "
                "Use phrases like 'as explained in the video,' 'as mentioned around,' or 'as demonstrated in the example at' to cite the timestamps and "
                "to help connect the response to the video content directly. Encourage students to revisit the video sections for a deeper understanding. "
                "Ensure your responses are tailored to the context of the video and focus on promoting an interactive and supportive learning environment."
                "Without any additonal text, return this answer."
            ),
        }
        self.full_transcript.append(system_prompt)
        
    def preprocess_question(self, question):
        tokens = word_tokenize(question.lower())
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)

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

    def generate_summary(self):
        try:
            with open('transcriptions.json', 'r') as file:
                transcriptions_data = json.load(file)
            summary_texts = []
            for entry in transcriptions_data:
                summary_texts.append(entry.get('summary', 'No summary available'))
            aggregated_summary = " ".join(summary_texts)
            
            prompt = (
                f"Please summarize the following information into two clear and concise paragraphs. The information is a from a video that the user is watching, and learning from. "
                f"The summary should highlight the key concepts, include important details, and provide any relevant examples to help the user understand. "
                f"Make sure the content is easy to follow and captures the main points. \n\n"
                f"{aggregated_summary}"
            )
            messages_to_send = [{"role": "user", "content": prompt}]
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_to_send,
                max_tokens=10000
            )

            ai_response = response.choices[0].message.content
            print("\nAI Summary Response:\n", ai_response)
            self.prompt_user()
        except Exception as e:
            print(f"An error occurred: {e}")

    
    def generate_ai_response(self, transcript, mode="audio"):
        self.stop_transcription()
        user_text = transcript.text
        processed_question = self.preprocess_question(user_text)
        relevant_context = self.get_relevant_context(processed_question)
        final_prompt = (
        f"This is the user question:\n{user_text}\n\n"
        f"These are the relevant contexts in order of relevancy:\n{relevant_context}\n\n"
        "Use the provided context to craft a well-structured, engaging, and listener-friendly response to the question. "
        "Ensure your answer is accurate and uses examples, references, or quotes from the context to provide clarity and depth. "
        "If possible, highlight specific sections or details from the context, using timestamps or references to guide the listener back to the source material. "
        "Always include timestamps from the provided context if they are available, and mention them explicitly in your response as it would help the student learn better."
        "For example, include phrases like 'as mentioned in the context,' 'as explained in the section around,' or 'as demonstrated in the example provided.' "
        "If the context does not fully address the question, rely on your knowledge to give a concise, efficient answer that remains accurate and informative. "
        "After completing the response, provide the YouTube link for the relevant timestamp as a separate line in the following format: "
        "'Relevant section: https://www.youtube.com/watch?v=mScpHTIi-kM&t=XXX' where 'XXX' is the most relevant timestamp in seconds. "
        "Prioritize clarity and engagement in your response, tailoring it to ensure it fits the tone and style expected by the audience. "
        "Return only the answer and the link with no additional commentary or text."
        )
        self.full_transcript.append({"role": "user", "content": user_text})
        messages_to_send = [
            self.full_transcript[0],  
            {"role": "user", "content": final_prompt}, 
        ]
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            max_tokens=10000
        )
        ai_response = response.choices[0].message.content
        self.full_transcript.append({"role": "assistant", "content": ai_response})
        if mode == "audio":
            asyncio.run(self.generate_audio(ai_response))
        else:
            print(f"\nAI Teacher: {ai_response}")
            self.prompt_user()

    async def generate_audio(self, text):
        print(f"\nAI Teacher: {text}")
        communicate = edge_tts.Communicate(text, "en-AU-NatashaNeural", rate="+20%")
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
            user_input = input(
                "\nEnter 1 to ask a question (voice mode), 2 to ask a question in text-mode, 3 to get a summary of the conversation/video, or 0 to finish the conversation: ")
            if user_input == "1":
                self.start_transcription()
            elif user_input == "2":
                question = input("\nPlease type your question: ")
                transcript = type("Transcript", (object,), {"text": question})()
                self.generate_ai_response(transcript, mode="text")
            elif user_input == "3":
                print("\nGenerating summary of the conversation...\n")
                self.generate_summary()
            elif user_input == "0":
                print("\nConversation Transcript: \n\n")
                self.full_transcript.pop(0)
                for entry in self.full_transcript:
                    print(f"{entry['role'].capitalize()}: {entry['content']}\n\n")
                print("\nThank you for the conversation!\n")
                self.cleanup()
                os._exit(1)
            else:
                print("Invalid input. Please enter 1, 2, or 0.")
                self.prompt_user()
        except KeyboardInterrupt:
            self.cleanup()
            print("\nProgram interrupted. Exiting.")
            os._exit(1)

    def cleanup(self):
        self.stop_transcription() 
        os._exit(1)
        
greeting = "Hey! How are you doing, How can I help you?"
ai_assistant = AI_Assistant()
asyncio.run(ai_assistant.generate_audio(greeting))

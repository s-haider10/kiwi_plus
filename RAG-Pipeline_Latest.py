from yt_dlp import YoutubeDL
import os
import argparse
import subprocess
import cv2
from scenedetect import detect, ContentDetector
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from chromadb.config import Settings
import os
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

openai_client = OpenAI(api_key="")
aai.settings.api_key = ""

def convert_to_seconds(timestamp):
    hours, minutes, seconds = map(int, timestamp.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

def generate_summary(text):
    openai_client = OpenAI(api_key="")
    aai.settings.api_key = ""
    system_prompt = """
            Your job is to analyze a text snippet, and analyze the text content, summarizing it in one line. Highlight the main ideas, key terms, and concepts being discussed. \n
            Without any additional text, return this summary.
    """
    response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}],
            max_tokens=10000
        )
    summary = response.choices[0].message.content
    return summary

def add_summaries_to_json(json_file): 
    with open(json_file, 'r') as f:
        transcriptions = json.load(f)
    for entry in transcriptions:
        full_text = " ".join([utterance["text"] for utterance in entry["transcription"]])
        summary = generate_summary(full_text)
        entry["summary"] = summary
    with open(json_file, 'w') as f:
        json.dump(transcriptions, f, indent=4)
    print(f"Summaries have been added and the file is updated.")

def transcribe_audio_in_chunks(audio):
    CHUNK_SIZE = 45 
    transcriptions = []
    chunks_dir = "audio_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    num_chunks = len(audio) // (CHUNK_SIZE * 1000) 
    for i in range(num_chunks + 1):
        start_time = i * CHUNK_SIZE * 1000 
        end_time = (i + 1) * CHUNK_SIZE * 1000
        chunk = audio[start_time:end_time]
        chunk_filename = os.path.join(chunks_dir, f"chunk_{i}.wav")
        chunk.export(chunk_filename, format="wav")
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            summarization=True,
            sentiment_analysis=True,
            entity_detection=True,
            speaker_labels=True,
            filter_profanity=True,
            language_detection=True
        )
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(chunk_filename)
    
        while transcript.status != aai.TranscriptStatus.completed:
            transcript = transcriber.transcribe(chunk_filename) 
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Error in chunk {i}: {transcript.error}")
        else:        
            transcriptions.append({
                "start_time": str(timedelta(milliseconds=start_time)),
                "end_time": str(timedelta(milliseconds=end_time)),
                "transcription": [  # Store each speaker's utterance separately
                    {"speaker": utterance.speaker, "text": utterance.text}
                    for utterance in transcript.utterances
                ]
            })
        os.remove(chunk_filename)
    with open('transcriptions.json', 'w') as json_file:
        json.dump(transcriptions, json_file, indent=4)
    print("Transcription completed and saved to 'transcriptions.json'.")

def perform_ocr(image_dir):

    def load_timestamps(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading timestamps from {json_path}: {e}")
            return {}
        
    def perform_ocr_with_preprocessing(image_dir, output_json_path, timestamps, max_text_length=500, language='eng'):
        ocr_results = []
        for image_file in os.listdir(image_dir):
            if image_file.endswith(".png"):
                image_path = os.path.join(image_dir, image_file)
                timestamp = timestamps.get(image_file, "00:00:00")
                try:
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                    text = pytesseract.image_to_string(thresh, lang=language).strip()
                    if len(text) > max_text_length:
                        text = text[:max_text_length] + "..." 
                
                    ocr_results.append({
                        "timestamp": timestamp,
                        "file_name": image_file,
                        "text": text
                    })
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        try:
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                json.dump(ocr_results, json_file, ensure_ascii=False, indent=4)
            print(f"OCR results saved to {output_json_path}")
        except Exception as json_error:
            print(f"Error saving JSON file: {json_error}")
        return ocr_results

    timestamps_json_path = os.path.join(image_dir, 'timestamps.json')
    output_json_path = 'ocr_results_with_timestamps_newest.json' 
    max_text_length = 50000
    timestamps = load_timestamps(timestamps_json_path)
    ocr_results = perform_ocr_with_preprocessing(image_dir, output_json_path, timestamps, max_text_length, language='eng')
    return None

def scene_detection(video_path):
    def format_timedelta(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    video_path = "downloaded_video.mp4"
    output_dir = "scenes"
    os.makedirs(output_dir, exist_ok=True)
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamps = {}
    for i, scene in enumerate(scenes):
        scene_start_frame = scene[0].get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start_frame)
        ret, frame = cap.read()
        if ret:
            scene_filename = f"scene_{i:03d}.png"
            cv2.imwrite(os.path.join(output_dir, scene_filename), frame)
            timestamp = scene_start_frame / fps
            timestamps[scene_filename] = format_timedelta(timestamp)
    cap.release()
    with open(os.path.join(output_dir, "timestamps.json"), "w") as f:
        json.dump(timestamps, f, indent=4)
    print(f"Scene detection complete. Selected scenes saved in '{output_dir}' folder.")
    print(f"Timestamps saved in '{output_dir}/timestamps.json'.")

    # input_folder = 'scenes'
    # output_folder = 'filtered_scenes'
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # def contains_person(image_path):
    #     img = cv2.imread(image_path)
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     results = model(img_rgb)  
    #     labels = results.names 
    #     person_detected = any(label == 'person' for label in results.names)
    #     return person_detected
    
    # for filename in os.listdir(input_folder):
    #     image_path = os.path.join(input_folder, filename)
    #     if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         print(f"Processing {filename}...")
    #         if contains_person(image_path):
    #             print(f"Person detected in {filename}, deleting...")
    #             os.remove(image_path)
    #         else:
    #             output_path = os.path.join(output_folder, filename)
    #             os.rename(image_path, output_path)

    # print("Processing complete!")

def download_video(url: str) -> str:
    ydl_opts = {
        'format': 'best[ext=mp4]', 
        'outtmpl': 'downloaded_video.mp4',
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    video_path = 'downloaded_video.mp4'
    print(f"Video downloaded successfully: {video_path}")
    return video_path

def extract_audio(video_path):
    audio_path = 'extracted_audio.wav'
    try:
        subprocess.call(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path])
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")
        return None
    return audio_path

def main():
    openai_client = OpenAI(api_key="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")
    aai.settings.api_key = "ceeca40708804bcfa78588a8b7b3349a"
    
    parser = argparse.ArgumentParser(description="Download a YouTube video.")
    parser.add_argument(
        '-url', '--url', 
        help='YouTube video URL. If not provided, the program will prompt for it interactively.',
        required=False
    )

    args = parser.parse_args()

    if args.url:
        video_url = args.url
    else:
        video_url = input("Please enter the YouTube video URL: ").strip()
    if not video_url:
        print("No URL provided. Exiting.")
        return
    
    video_path = download_video(video_url)
   
    audio_path = extract_audio(video_path)
   
    scene_detection(video_path)
    
    perform_ocr('scenes')

    audio = AudioSegment.from_wav(audio_path)

    transcribe_audio_in_chunks(audio)

    add_summaries_to_json("transcriptions.json")

    with open('scenes/timestamps.json', 'r') as f:
        timestamps_images = json.load(f)
        
    timestamps_in_seconds = {image: convert_to_seconds(time) for image, time in timestamps_images.items()}

    with open('transcriptions.json', 'r') as f:
        transcription_data = json.load(f)

    for segment in transcription_data:
        start_time_sec = convert_to_seconds(segment["start_time"])
        end_time_sec = convert_to_seconds(segment["end_time"])
        image_names = [
            image for image, time_sec in timestamps_in_seconds.items()
            if start_time_sec <= time_sec <= end_time_sec
        ]
        segment["image_names"] = image_names

    with open('transcriptions.json', 'w') as f:
        json.dump(transcription_data, f, indent=4)
    
    transcription_data = 'transcriptions.json'

    client_capstone = chromadb.PersistentClient(path="chromadb")
    collection_transcriptions_clip = client_capstone.create_collection(name="Capstone_Kiwi_clip")
    collection_transcriptions_bert = client_capstone.create_collection(name="Capstone_Kiwi_bert")
    
    with open(transcription_data, 'r') as file:
        transcription_data = json.load(file)
    data = transcription_data
    model_name = "openai/clip-vit-base-patch32"


    clip_model = CLIPModel.from_pretrained(model_name)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = clip_model.to(device)


    bert_model_name = "bert-base-uncased"
    bert_model = AutoModel.from_pretrained(bert_model_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)

    def embed_text_bert(text):
        inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            text_embedding = outputs.last_hidden_state[:, 0, :]
        return text_embedding.squeeze().cpu().numpy()
    
    def embed_text(text):
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_embeddings = clip_model.get_text_features(**text_inputs)
        return text_embeddings.squeeze().cpu().numpy()
    
    embeddings = []
    for item in transcription_data:
        summary_text = item['summary']
        embedding = embed_text(summary_text)
        embeddings.append(embedding)
    
    for i, item in enumerate(transcription_data):
        summary_text = item['summary']
        collection_transcriptions_clip.add(
            documents=str(item),
            embeddings=[embeddings[i]],
            metadatas=None,
            ids=[str(i)]
        )

    bert_embeddings = []
    for item in transcription_data:
        summary_text = item['summary']
        embedding = embed_text_bert(summary_text)
        bert_embeddings.append(embedding)
    
    for i, item in enumerate(transcription_data):
        collection_transcriptions_bert.add(
            documents=str(item),
            embeddings=[bert_embeddings[i]],
            metadatas=None,
            ids=[str(i)]
        )
    
    print("Data successfully added to collection_transcriptions.")
    print('Now we can initiate our audio LLM pipeline.')

if __name__ == "__main__":
    main()

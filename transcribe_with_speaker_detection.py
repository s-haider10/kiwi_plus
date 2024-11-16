import assemblyai as aai
from datetime import timedelta
from pydub import AudioSegment
import json
import os
from openai import OpenAI
import json

openai_client = OpenAI(api_key="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")
aai.settings.api_key = "ceeca40708804bcfa78588a8b7b3349a"

FILE_URL = "audio_files/extracted_audio.wav"
CHUNK_SIZE = 45 
audio = AudioSegment.from_wav(FILE_URL)
chunks_dir = "audio_chunks"

os.makedirs(chunks_dir, exist_ok=True)
transcriptions = []

def transcribe_audio_in_chunks(audio):
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

transcribe_audio_in_chunks(audio)
with open('transcriptions.json', 'w') as json_file:
    json.dump(transcriptions, json_file, indent=4)

print("Transcription completed and saved to 'transcriptions.json'.")

def generate_summary(text):
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

add_summaries_to_json("transcriptions.json")

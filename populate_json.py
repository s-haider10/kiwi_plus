from openai import OpenAI
import json
from datetime import timedelta

openai_client = OpenAI(api_key="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")

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


def convert_to_seconds(timestamp):
    hours, minutes, seconds = map(int, timestamp.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

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
    




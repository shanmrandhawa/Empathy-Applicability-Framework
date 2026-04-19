# !pip install openai
import pandas as pd
import openai
import json
import openai
from openai import OpenAI

# Set your OpenAI API key
api_key_personal = ".."

import httpx

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

client = OpenAI(api_key=api_key_personal, http_client=CustomHTTPClient())


csv_file_path = "data_queries.csv"  # Replace with your file path
df = pd.read_csv(csv_file_path)
df.rename(columns={df.columns[0]: 'patient_query'}, inplace=True)

# Full annotation schema from the JSON document
ANNOTATION_SCHEMA = {
    "instruction": "Annotate emotional reactions in general health queries based on the following schema. For each query, return the matching subcategories. Think logically and ensure to revisit your annotation for each query",
    "definitions": {
        "Emotional Reactions": {
            "description": "Expressions of warmth, compassion, concern, or similar feelings conveyed by a doctor in response to a patient's query.",
            "categories": [
                {
                    "name": "Purely Factual Medical Queries",
                    "description": "The patient requests specific medical information, including explanations of medical concepts, without emotional distress or underlying distressing uncertainty.",
                    "examples": ["What is the use of Tylenol?", "Is it possible to outgrow a seafood allergy?"],
                    "class" : "Emotional Reactions Not Applicable"
                },
                {
                    "name": "General Health Management Without Emotional Involvement",
                    "description": "The patient seeks guidance on health management, follows up on prior advice, or requests basic guidance on minor health issues, without expressing emotional distress or underlying distressing uncertainty. Here the guidance is on what the patient should do.",
                    "examples": ["I'm managing diabetes with insulin. How often should I check my blood sugar levels?", "I have swelling in my ankle after a long walk. Should I be concerned?"],
                    "class" : "Emotional Reactions Not Applicable"
                },
                {
                    "name": "Diagnosis Requests with Neutral Symptom Descriptions",
                    "description": "The patient describes symptoms neutrally without expressing emotional distress or underlying distressing uncertainty. Here the request is about asking what the doctor thinks the issue is.",
                    "examples": ["I have intermittent knee pain from working out. How would I know if I tore cartilage?", "Hello. I am having pain in my jaw area, immediately in front of my left ear."],
                    "class" : "Emotional Reactions Not Applicable"
                },
                {
                    "name": "Hypothetical Medical Queries Without Emotional Concern",
                    "description": "The patient inquires about hypothetical situations without emotional involvement.",
                    "examples": ["If someone has XYZ symptoms, what might be the cause?", "What would happen if a person skipped their medication?"],
                    "class" : "Emotional Reactions Not Applicable"
                },
                {
                    "name": "Seriousness of Symptoms",
                    "description": "The patient describes symptoms that suggest a life-threatening or chronic health condition significantly impacting long-term health or quality of life. This includes diseases like cancer, heart disease, mental health issues, or chronic conditions leading to disability.The symptoms suggest a life-threatening or serious health condition that could significantly impact long-term health or quality of life.",
                    "examples": ["My father has been having severe chest pains and shortness of breath. Could it be a heart attack?", "I've been experiencing numbness and weakness in my limbs for months."],
                    "class" : "Emotional Reactions Applicable"
                },
                {
                    "name": "Severe Negative Emotion Expressed",
                    "description": "The patient explicitly states intense emotions such as fear, frustration, or anger regarding their health.",
                    "examples": ["I feel depressed and anxious like never before. I cannot sleep at night.", "I'm terrified about my recent diagnosis of cancer."],
                    "class" : "Emotional Reactions Applicable"
                },
                {
                    "name": "Underlying Negative Emotional State Inferred",
                    "description": "The patient implies emotional distress that isn't explicitly stated but can be inferred from their tone or descriptions, such as subtle signs of emotional worry, frustration, or distress about delays or underlying distressing uncertainties. Focus on emotional worry, not the medical concern.",
                    "examples": ["I am starting to get a little alarmed by this spotting after ovulation. Is this cause for concern?", "I need to be a bit more at ease after what I read about diabetic enteropathy."],
                    "class" : "Emotional Reactions Applicable"
                },
                {
                    "name": "Concern Severity for Close Relations",
                    "description": "The patient is asking on behalf of someone with whom they share a close, protective relationship, implying heightened emotional concern.",
                    "examples": ["Hello, I am the mother of a five-year-old. He has a small lump that hasn't gone away.", "My son recently started daycare and has gotten sick. His fever was 102.9. Should I take him to the hospital?"],
                    "class" : "Emotional Reactions Applicable"
                }
            ]
        }
    },
    "output_format": "json",
    "example_query": {
    "query": "I'm scared and plan on taking my son to the doctor. Should I be overly worried?",
    "annotations": [
      {
        "subcategories": [
          {
            "name": "Severe Negative Emotion Expressed",
          },
          {
            "name": "Concern Severity for Close Relations",
          }
        ],
        "class": "Emotional Reactions Applicable",
        "reason": "The patient explicitly expresses fear regarding their son's health and shows heightened emotional concern for a close relation."
      }
    ]
  }
}

# Function to annotate a query using OpenAI API
def annotate_query(query):
    try:
        # Sending schema and query to the API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert annotator following a structured schema."},
                {"role": "user", "content": json.dumps({
                    "instructions": ANNOTATION_SCHEMA,
                    "query": query
                })}
            ]
        )
        # Extract and parse the assistant's response
        annotation = response.choices[0].message.content
        print(annotation)
        return annotation
    except Exception as e:
        print(f"Error annotating query: {query}\n{str(e)}")
        return False

# Function to parse and check applicability
def check_applicability(json_output):
  if "Emotional Reactions Applicable" in json_output:
      return "Applicable"
  elif "Not" in json_output:
    return "Not Applicable"
  return "Error"

import time

# Process the DataFrame in chunks of 50 rows
chunk_size = 500
output_file_prefix = "emotional_annotations_chunk"

for start in range(0, len(df), chunk_size):
    end = start + chunk_size
    print(f"Processing rows {start} to {end-1}")

    try:
        for index, row in df.iloc[start:end].iterrows():
            print(f"Processing row {index}")
            annotation = annotate_query(row['patient_query'])
            df.at[index, 'raw'] = annotation
            applicability = check_applicability(annotation)
            if applicability != "Error":
              df.at[index, 'applicability'] = applicability
              time.sleep(1)  # To avoid rate limits
            else:
              df.at[index, 'applicability'] = "Error"
              time.sleep(1)  # To avoid rate limits
    except Exception as e:
        print(f"Error during chunk processing: {e}")
        time.sleep(30)
        pass

    # Save the results to a CSV file after processing each chunk
    chunk_output_path = f"{output_file_prefix}_{end-1}.csv"
    df.iloc[:end].to_csv(chunk_output_path, index=False)
    print(f"Saved results to {chunk_output_path}")

# Save the final updated DataFrame
df.to_csv("final_emotional_annotations.csv", index=False)
print("All chunks processed and results saved.")

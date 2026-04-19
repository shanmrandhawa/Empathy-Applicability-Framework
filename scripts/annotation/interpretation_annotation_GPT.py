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
ANNOTATION_SCHEMA = SCHEMAS = {
        "instruction": "Annotate interpretations in general health queries based on the following schema. For each query, return the matching subcategories. Think logically and ensure to revisit your annotation for each query",
        "definitions": {
            "Interpretations": {
                "description": "Interpretations refer to the communication of an understanding of the patient’s feelings (explicit or implied) and/or experiences (contextual factors) inferred from their query. It's about recognizing and articulating what the patient is feeling and why, based on their situation, concerns, and history.",
                "categories": [
                    {
                        "name": "Expression of Feelings (Explicit or Implicit)",
                        "description": "The patient expresses emotions directly or implies them through language or tone. This includes feelings such as fear, anxiety, frustration, sadness, or hopelessness.",
                        "examples": [
                            "I'm really scared about these chest pains.",
                            "I'm frustrated because my symptoms aren't improving.",
                            "I guess I have to accept this is how things will be now.",
                            "Nothing seems to be helping.",
                            "I don't know what to do anymore."
                        ],
                        "class": "Interpretations Applicable"
                    },
                    {
                        "name": "Sharing of Experiences or Contextual Factors Affecting Emotional State and Well being",
                        "description": "The patient shares personal experiences, contextual factors, or circumstances that influence their health and emotional state. These include social, environmental, or personal situations, beyond medical concerns, that affect their emotional state.",
                        "examples": [
                            "With my father’s illness and financial stress, I’m feeling overwhelmed.",
                            "I've been under a lot of pressure at work, and now I'm having trouble sleeping.",
                            "Ever since the accident, I can't stop thinking about what happened.",
                            "I recently moved to a different state, haven't found a general practitioner, and haven't paid my high deductible for the year."
                        ],
                        "class": "Interpretations Applicable"
                    },
                    {
                        "name": "Expressions of Distressing Uncertainty About Health or Treatment",
                        "description": "Uncertainties, confusion, or mistrust expressed by a patient about their health status, treatment, or future that lead to significant emotional distress.This includes statements involving questions about prognosis, treatment effectiveness, or doubt about potential outcomes, specifically when accompanied by explicit or implicit signs of emotional distress. The focus should remain on uncertainty that reflects or suggests underlying emotional distress in the patient.",
                        "examples": [
                            "I’m not sure if this treatment is really working for me, and it’s making me anxious.",
                            "Is this something that sounds like I should consider doing? I’m so confused about what’s right.",
                            "I feel lost. Should my wife also get examined?",
                            "Do you think there’s any hope for me after trying this?"
                        ],
                        "class": "Interpretations Applicable"
                    },
                    {
                        "name": "Symptoms Significantly Affecting Emotional Well-being or Daily Life",
                        "description": "The patient describes symptoms that significantly impact their emotional well-being or daily functioning, and they express or imply emotional distress because of these symptoms. The key is the emotional impact of the symptoms, not just the symptoms themselves.",
                        "examples": [
                            "These migraines are making it impossible to enjoy my hobbies.",
                            "I'm so tired all the time that I can't take care of my kids properly.",
                            "My symptoms have been affecting my job for months.",
                            "The pain is getting worse every day, and it's really wearing me down."
                        ],
                        "class": "Interpretations Applicable"
                    },
                    {
                        "name": "Straightforward Medical Queries Lacking Emotion, Distressing Uncertainty, and Context",
                        "description": "The patient requests specific medical information or explanations of medical concepts without expressing emotional distress, underlying distressful uncertainty or providing context (social, environmental, or personal situations) implying an emotional state. These queries are strictly informational and lack emotional or experiential elements requiring interpretation.",
                        "examples": [
                            "What is the use of Tylenol?",
                            "Hello doctor, I would like to get an opinion regarding the attached chest radiograph. I wish to know if there are any abnormalities like scarring."
                        ],
                        "class": "Interpretations Not Applicable"
                    },
                    {
                        "name": "General health management requests Without Emotion, Context, and Distressing Uncertainty",
                        "description": "The patient seeks guidance on health management, follows up on prior advice, or requests basic guidance on minor health issues without expressing emotional distress, underlying distressful uncertainty, or providing contextual factors (social, environmental, or personal situations) that imply an emotional state.Additionally, they do includes personal medical context, such as test results, medications taken, and previous medical consultations. Here the guidance is on what the patient should do.",
                        "examples": [
                            "I have intermittent knee pain from working out. How would I know if I tore cartilage?",
                            "I had an X-ray for a fracture; should it be strapped or cast right away?",
                            "Hi, my husband is 39, and his SGPT and SGOT levels in a recent test were 101 and 98 respectively. His triglycerides are 280, which is high. His height is 168 cm and weight is 79 kg. What does a rise in these values indicate? What precautions should he take?"
                        ],
                        "class": "Interpretations Not Applicable"
                    },
                                        {
                        "name": "Diagnosis Requests with Neutral Symptom Descriptions Lacking Distressing Uncertainty and Context",
                        "description": "The patient describes symptoms neutrally without expressing emotional distress or underlying distressful uncertainty. They provide necessary details without implying feelings or contextual factors (social, environmental, or personal situations) that need acknowledgment. These descriptions are straightforward and lack emotional or experiential content requiring interpretation. Here the request is about asking what the doctor thinks the issue is.",
                        "examples": [
                            "I have swelling in my ankle after a long walk. Should I be concerned?",
                            "Hello doctor, I am suffering from pain in my mouth. It feels like sensitivity pain. I cannot say it is pain exactly; it is irritating a lot. No pain in teeth. It feels like itching in my gums (middle of the teeth). Please tell me what I can do."
                        ],
                        "class": "Interpretations Not Applicable"
                    },
                                        {
                        "name": "Hypothetical Medical Queries with no Emotions, Context, and Distressing Uncertainty",
                        "description": "The patient inquires about hypothetical situations or general medical information without expressing or implying personal feelings or contextual factors (social, environmental, or personal situations) that need acknowledgment. These queries are theoretical and lack emotional or experiential aspects requiring interpretation.",
                        "examples": [
                            "If someone has XYZ symptoms, what might be the cause?",
                            "What would happen if a person skipped their medication?"
                        ],
                        "class": "Interpretations Not Applicable"
                    }
                ]
            }
        },
        "output_format": "json",
        "example_query": {
            "query": "I’m not sure if this treatment is really working for me.",
            "annotations": [
                {
                    "subcategories": [
                        {
                            "name": "Expressions of Uncertainty About Health or Treatment",
                        }
                    ],
                    "class": "Interpretations Applicable",
                    "reason": "The patient explicitly expresses doubt about the effectiveness of the treatment, which requires interpretation."
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
  if "Interpretations Applicable" in json_output:
      return "Applicable"
  elif "Not" in json_output:
    return "Not Applicable"
  return "Error"

import time

# Process the DataFrame in chunks of 50 rows
chunk_size = 500
output_file_prefix = "interpertations_annotations_chunk"

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
df.to_csv("final_interpertations_annotations.csv", index=False)
print("All chunks processed and results saved.")

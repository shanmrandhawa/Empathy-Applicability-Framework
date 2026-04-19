# !pip install openai
import pandas as pd
import openai
import json
import openai
from openai import OpenAI

# Set your OpenAI API key
api_key_personal = ""

import httpx

class CustomHTTPClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
        super().__init__(*args, **kwargs)

client = OpenAI(api_key=api_key_personal, http_client=CustomHTTPClient())


csv_file_path = "EA_test.csv"
df = pd.read_csv(csv_file_path)

df.rename(columns={df.columns[0]: 'patient_query'}, inplace=True)

# Minimal instruction (without framework)
ANNOTATION_SCHEMA = {
    "instruction": "Read the patient query and decide whether emotional reactions are necessary in the response. "
                   "Emotional reactions refer to the expressions of warmth, compassion, concern, or similar feelings "
                   "conveyed by a doctor in response to a patient's query. "
                   "If emotional reactions are necessary, mark it as 'Emotional Reactions Applicable'. "
                   "If not, mark it as 'Emotional Reactions Not Applicable'. Think carefully and be consistent.",
    "output_format": "json"
}

# Function to annotate a query using OpenAI API
def annotate_query(query):
    try:
        # Sending schema and query to the API
        response = client.chat.completions.create(
            model="o1",
            messages=[
                {"role": "system", "content": "You are an expert annotator."},
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
chunk_size = 50
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
              df.at[index, 'EA_applicability'] = applicability
              time.sleep(1)  # To avoid rate limits
            else:
              df.at[index, 'EA_applicability'] = "Error"
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
df.to_csv("emotional_annotations_o1_zeroshot.csv", index=False)
print("All chunks processed and results saved.")


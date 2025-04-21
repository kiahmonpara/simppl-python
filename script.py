import base64
import os
import time
from google import genai
from google.genai import types
import json
from datetime import datetime

def generate(): 
    client = genai.Client(
        api_key='AIzaSyDVFjlgmZdy9a7bmmg3-FVA2eE_V3qp2tM',
    )
    input_file = "output.json"
    output_file = "output_new.json"
    
    try:
        with open(input_file, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file {input_file} is not valid JSON.")
        return
    
    all_results = []
    
    try:
        with open(output_file, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results from {output_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Starting fresh output file: {output_file}")
    
    batch_size = 10
    total_objects = len(all_data)
    processed_count = 0
    
    if all_results:
        for result in all_results:
            if isinstance(result, dict) and "batch_info" in result:
                processed_count = max(processed_count, result["batch_info"]["end_index"])
    
    print(f"Starting from object {processed_count} of {total_objects}")
    
    for start_index in range(processed_count, total_objects, batch_size):
        end_index = min(start_index + batch_size, total_objects)
        current_batch = all_data[start_index:end_index]
        
        print(f"Processing batch {start_index}-{end_index-1} ({len(current_batch)} objects)")
        
        final_prompt = f"""Analyze the Reddit post data provided in the file which is in json format and extract insights specifically related to harmful content. Identify and categorize posts based on the following criteria:

Harmful Content Detection:

Posts containing bad words (list detected words).

Posts with potential misinformation (list misleading phrases).

Biased or politically charged content (list detected political terms).

Potentially offensive content (explain why it might be offensive).

Indicate whether the post was banned or not.

Post Metadata & Engagement Analysis:

Post ID: Unique identifier of the post.

Author Fullname: The username of the post's author.

Subreddit: The subreddit where the post was uploaded.

Engagement Stats: Extract upvote count, downvote count, comments count, and any other engagement metrics.

Upvote/Downvote Ratio: Calculate and include the ratio.

Insightful Commentary:

Summary (2-3 lines): High-level overview of the trends observed in harmful content.

Specific Findings: Provide an analysis of detected phrases in relation to real-world events, biases, or misinformation patterns.

Attraction Factor: Explain what makes these posts engaging or viral (e.g., emotional appeal, controversial nature).

Extra Context: Additional observations on how such posts impact discussions or social sentiment.

Ensure all posts are included, with structured and comprehensive insights, dont put all the information in a single property , every insight i need has an assigned property.
{json.dumps(current_batch)}
"""
        
        try:
            model = "gemini-2.0-flash-lite"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=final_prompt)],
                ),
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text="""{
  "summary": "Analysis identified that certain posts within the Reddit dataset used bad words, engaged in politically charged discourse, and had the potential to spread misinformation. All identified insights about each post are assigned to the respective attributes to ensure precise insights. There was minimal user engagement with the potential harmful content, as each post is designed to adhere to the designated properties, with every extraction and insight following the well-formed structure and clarity."
}"""),
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="""INSERT_INPUT_HERE""")],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "Specific Findings": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            items=genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        ),
                        "Attraction": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            items=genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        ),
                        "Post": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            items=genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                required=["ID", "user", "Bad words", "Political words", "Potential Misinfo", "banned", "subreddit", "extra"],
                                properties={
                                    "ID": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                    ),
                                    "user": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                    ),
                                    "Bad words": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                    "Political words": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                    "Potential Misinfo": genai.types.Schema(
                                        type=genai.types.Type.ARRAY,
                                        items=genai.types.Schema(
                                            type=genai.types.Type.STRING,
                                        ),
                                    ),
                                    "banned": genai.types.Schema(
                                        type=genai.types.Type.BOOLEAN,
                                    ),
                                    "subreddit": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                    ),
                                    "extra": genai.types.Schema(
                                        type=genai.types.Type.STRING,
                                    ),
                                },
                            ),
                        ),
                        "summary": genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    },
                ),
            )
            
            batch_chunks = []
            
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                batch_chunks.append(chunk.text)
                print(chunk.text, end="")
            
            batch_result = "".join(batch_chunks)

            try:
                batch_json = json.loads(batch_result)
                batch_json["batch_info"] = {
                    "start_index": start_index,
                    "end_index": end_index,
                    "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_results.append(batch_json)
            except json.JSONDecodeError:
                print(f"\nError: Invalid JSON response for batch {start_index}-{end_index-1}")
                all_results.append({
                    "batch_info": {
                        "start_index": start_index,
                        "end_index": end_index,
                        "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": "Invalid JSON response"
                    },
                    "raw_response": batch_result
                })

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            
            print(f"\nProcessed batch {start_index}-{end_index-1}, saved results to {output_file}")
            
            if end_index < total_objects:
                print(f"Waiting 4 seconds before next batch...")
                time.sleep(4)
                
        except Exception as e:
            print(f"\nError processing batch {start_index}-{end_index-1}: {str(e)}")
            error_info = {
                "batch_info": {
                    "start_index": start_index,
                    "end_index": end_index,
                    "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e)
                }
            }
            all_results.append(error_info)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            
            print(f"Saved progress up to error at batch {start_index}-{end_index-1}")

            if end_index < total_objects:
                print(f"Waiting 4 seconds before next batch...")
                time.sleep(4)
    
    print(f"Processing complete. All results saved to {output_file}")

if __name__ == "__main__":
    generate()
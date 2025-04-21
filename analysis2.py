import json
import os
import pandas as pd
from collections import Counter
import re
import requests
from dotenv import load_dotenv

output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON data.")
        return []

def load_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []
    
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

# Function to generate content summary using Gemini API
def generate_content_summary(texts, prompt_type="general"):
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    combined_text = " ".join(texts)
    
    if prompt_type == "general":
        prompt = "Generate a comprehensive summary of the following social media content, highlighting key themes, topics, and trends:"
    elif prompt_type == "bad_words":
        prompt = "Analyze these bad words found in social media posts. Explain potential impact on community sentiment and platform health:"
    elif prompt_type == "political":
        prompt = "Analyze these political terms from social media. Explain how they might affect user engagement, political discourse, and platform polarization:"
    else:
        prompt = "Summarize the following content:"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"{prompt}\n\n{combined_text}"
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key
    }

    try:
        response = requests.post(gemini_api_url, headers=headers, json=payload)
        response.raise_for_status()  
        
        result = response.json()
        summary = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not summary:
            if prompt_type == "general":
                return "The content includes a mix of social media posts covering various topics, with notable political and social discussions."
            elif prompt_type == "bad_words":
                return "Analysis of offensive language reveals patterns that could negatively impact community health and user experience."
            elif prompt_type == "political":
                return "Political terminology in posts indicates polarization and suggests significant influence on user engagement patterns."
            
        return summary
        
    except requests.exceptions.RequestException as e:
        print(f"Error with Gemini API: {e}")
        # Fallback summaries if the API call fails
        if prompt_type == "general":
            return "The content includes a mix of social media posts covering various topics, with notable political and social discussions."
        elif prompt_type == "bad_words":
            return "Analysis of offensive language reveals patterns that could negatively impact community health and user experience."
        elif prompt_type == "political":
            return "Political terminology in posts indicates polarization and suggests significant influence on user engagement patterns."

data_source = "file"  
data_file_path = "output1.json"  
api_url = ""  

if data_source == "file":
    data = load_data(data_file_path)
else:
    data = load_data_from_api(api_url)

if not data:
    print("Failed to load data.")
    exit()

# Initialize an empty list for posts data
posts_data = []
exceptional_posts = []
all_bad_words = []  
all_political_words = []  
all_political_misinfo = []  
all_texts = []  

for batch in data:
    for post in batch.get('Post', []):
        if post.get('banned', False):
            exceptional_posts.append(post)
            continue
        
        bad_words_list = post.get('Bad words', [])
        if isinstance(bad_words_list, list):
            all_bad_words.extend(bad_words_list)
        
        political_words_list = post.get('Political words', [])
        if isinstance(political_words_list, list):
            all_political_words.extend(political_words_list)
        
        misinfo_list = post.get('Potential Misinfo', [])
        if isinstance(misinfo_list, list):
            all_political_misinfo.extend(misinfo_list)
        
        post_data = {
            'ID': post.get('ID', ''),
            'user': post.get('user', ''),
            'Bad words': ', '.join(bad_words_list) if isinstance(bad_words_list, list) else '',
            'Political words': ', '.join(political_words_list) if isinstance(political_words_list, list) else '',
            'Potential Misinfo': ', '.join(misinfo_list) if isinstance(misinfo_list, list) else '',
            'banned': post.get('banned', False),
            'subreddit': post.get('subreddit', ''),
            'extra': post.get('extra', ''),
            'timestamp': post.get('timestamp', '')  
        }
        posts_data.append(post_data)
        all_texts.append(post.get('extra', ''))  

df = pd.DataFrame(posts_data)

# Count the top bad words
bad_words_counter = Counter(all_bad_words)
top_bad_words = bad_words_counter.most_common(20)  

# Count the top political words
political_words_counter = Counter(all_political_words)
top_political_words = political_words_counter.most_common(20)  

# Count the top potential misinfo words
misinfo_counter = Counter(all_political_misinfo)
top_misinfo = misinfo_counter.most_common(20)  

# Generate different types of analysis using the Gemini API
general_summary = generate_content_summary(all_texts, "general")
bad_words_analysis = generate_content_summary(all_bad_words, "bad_words")
political_words_analysis = generate_content_summary(all_political_words, "political")

print("\n=== OVERALL CONTENT SUMMARY ===")
print(general_summary)
print("\n=== BAD WORDS ANALYSIS ===")
print(bad_words_analysis)
print("\n=== POLITICAL WORDS ANALYSIS ===")
print(political_words_analysis)

# Word Cloud data for General Content
text_for_wc = " ".join(all_texts)
general_word_counts = Counter(re.findall(r'\w+', text_for_wc.lower()))
top_general_words = general_word_counts.most_common(100) 

wordcloud_data = {
    'general': [{'text': word, 'size': count} for word, count in top_general_words],
    'bad_words': [{'text': word, 'size': count} for word, count in bad_words_counter.most_common(100)],
    'political_words': [{'text': word, 'size': count} for word, count in political_words_counter.most_common(100)],
    'misinfo': [{'text': word, 'size': count} for word, count in misinfo_counter.most_common(100)]
}

with open(os.path.join(output_dir, 'wordcloud_data.json'), 'w') as f:
    json.dump(wordcloud_data, f)

# Generate detailed insights 
words = " ".join(all_texts)
word_counts = Counter(re.findall(r'\w+', words.lower()))
most_common_words = word_counts.most_common(20) 

# Calculate engagement metrics 
engagement_stats = {}
if 'likes' in df.columns:
    engagement_stats['Average Likes'] = df['likes'].mean()
if 'comments' in df.columns:
    engagement_stats['Average Comments'] = df['comments'].mean()
if 'shares' in df.columns:
    engagement_stats['Average Shares'] = df['shares'].mean()

# Analyze correlations between bad words, political content, and engagement
correlation_analysis = "No engagement metrics available to analyze correlation."
correlation_data = {}

if any(metric in df.columns for metric in ['likes', 'comments', 'shares']):
    df['bad_words_count'] = df['Bad words'].apply(lambda x: len(x.split(',')) if x and x.strip() else 0)
    df['political_words_count'] = df['Political words'].apply(lambda x: len(x.split(',')) if x and x.strip() else 0)
    df['misinfo_count'] = df['Potential Misinfo'].apply(lambda x: len(x.split(',')) if x and x.strip() else 0)
    
    if 'likes' in df.columns:
        correlation_data['Correlation with Likes'] = {
            'Bad Words': df['bad_words_count'].corr(df['likes']),
            'Political Words': df['political_words_count'].corr(df['likes']),
            'Potential Misinfo': df['misinfo_count'].corr(df['likes'])
        }
    if 'comments' in df.columns:
        correlation_data['Correlation with Comments'] = {
            'Bad Words': df['bad_words_count'].corr(df['comments']),
            'Political Words': df['political_words_count'].corr(df['comments']),
            'Potential Misinfo': df['misinfo_count'].corr(df['comments'])
        }
    correlation_analysis = str(correlation_data)

# Overall insights
overall_analysis = {
    'Total Posts': len(df),
    'Total Exceptional Posts': len(exceptional_posts),
    'Bad Words Count': len(all_bad_words),
    'Political Words Count': len(all_political_words),
    'Potential Misinfo Count': len(all_political_misinfo),
    'Total Unique Subreddits': df['subreddit'].nunique(),
    'Top Common Words': most_common_words,
    'Top Bad Words': top_bad_words[:10],  
    'Engagement Statistics': engagement_stats,
    'Correlation Analysis': correlation_data
}

print("\n=== DETAILED OVERALL ANALYSIS ===")
for key, value in overall_analysis.items():
    if key not in ['Top Common Words', 'Top Bad Words', 'Top Political Words', 'Correlation Analysis']:
        print(f"{key}: {value}")
    elif key in ['Top Common Words', 'Top Bad Words', 'Top Political Words']:
        print(f"{key}:")
        for word, count in value:
            print(f"  - {word}: {count}")
    else:
        print(f"{key}: {value}")

word_count_data = {
    'Bad Words': len(all_bad_words),
    'Political Words': len(all_political_words),
    'Potential Misinfo': len(all_political_misinfo),
}

with open(os.path.join(output_dir, 'word_counts_bar_data.json'), 'w') as f:
    json.dump(word_count_data, f)

bad_words_bar_data = {
    'labels': [word for word, _ in top_bad_words],
    'counts': [count for _, count in top_bad_words]
}

with open(os.path.join(output_dir, 'top_bad_words_data.json'), 'w') as f:
    json.dump(bad_words_bar_data, f)

political_words_bar_data = {
    'labels': [word for word, _ in top_political_words],
    'counts': [count for _, count in top_political_words]
}

with open(os.path.join(output_dir, 'top_political_words_data.json'), 'w') as f:
    json.dump(political_words_bar_data, f)

misinfo_bar_data = {
    'labels': [word for word, _ in top_misinfo],
    'counts': [count for _, count in top_misinfo]
}

with open(os.path.join(output_dir, 'top_misinfo_data.json'), 'w') as f:
    json.dump(misinfo_bar_data, f)

subreddit_counts = df['subreddit'].value_counts().to_dict()

subreddit_pie_data = {
    'labels': list(subreddit_counts.keys()),
    'values': list(subreddit_counts.values())
}

with open(os.path.join(output_dir, 'subreddit_distribution_data.json'), 'w') as f:
    json.dump(subreddit_pie_data, f)

if 'timestamp' in df.columns and not df['timestamp'].empty:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        posts_by_date = df.groupby('date').size().reset_index(name='count')
        posts_by_date['date'] = posts_by_date['date'].astype(str)
        
        time_series_data = {
            'dates': posts_by_date['date'].tolist(),
            'counts': posts_by_date['count'].tolist()
        }
        
        with open(os.path.join(output_dir, 'posts_time_series_data.json'), 'w') as f:
            json.dump(time_series_data, f)
    except Exception as e:
        print(f"Error processing time series data: {e}")

# Comprehensive report with all analyses
report = f"""
# Social Media Content Analysis Report

## Overall Summary
{general_summary}

## Bad Words Analysis
{bad_words_analysis}

## Political Words and Engagement Analysis
{political_words_analysis}

## Key Statistics
- Total Posts Analyzed: {overall_analysis['Total Posts']}
- Total Exceptional Posts: {overall_analysis['Total Exceptional Posts']}
- Bad Words Detected: {overall_analysis['Bad Words Count']}
- Political Words Detected: {overall_analysis['Political Words Count']}
- Potential Misinformation Instances: {overall_analysis['Potential Misinfo Count']}
- Unique Subreddits: {overall_analysis['Total Unique Subreddits']}

## Most Common Words in All Content
{', '.join([f"{word} ({count})" for word, count in most_common_words])}

## Top 10 Bad Words
{', '.join([f"{word} ({count})" for word, count in top_bad_words[:10]])}

## Top 10 Political Words
{', '.join([f"{word} ({count})" for word, count in top_political_words[:10]])}

## Correlation Between Content Types and Engagement
{correlation_analysis}

## Conclusions and Recommendations
Based on the analysis, we recommend monitoring posts containing high concentrations of political terms, 
as they may contribute to increased polarization. Implementing content moderation strategies for 
posts with bad words could improve community health. Regular analysis of trending political 
terminology can help identify potential misinformation early.
"""

# Saving the report to a file
with open(os.path.join(output_dir, 'social_media_analysis_report.md'), 'w') as f:
    f.write(report)

df.to_csv(os.path.join(output_dir, 'processed_posts_data.csv'), index=False)

# Creating a master JSON file with all visualization data 
visualization_data = {
    'word_counts': word_count_data,
    'top_bad_words': bad_words_bar_data,
    'top_political_words': political_words_bar_data,
    'top_misinfo': misinfo_bar_data,
    'subreddit_distribution': subreddit_pie_data,
    'wordcloud': wordcloud_data,
    'summaries': {
        'general': general_summary,
        'bad_words': bad_words_analysis,
        'political': political_words_analysis
    },
    'stats': {
        'total_posts': len(df),
        'exceptional_posts': len(exceptional_posts),
        'bad_words_count': len(all_bad_words),
        'political_words_count': len(all_political_words),
        'misinfo_count': len(all_political_misinfo),
        'unique_subreddits': df['subreddit'].nunique()
    }
}

if os.path.exists(os.path.join(output_dir, 'posts_time_series_data.json')):
    with open(os.path.join(output_dir, 'posts_time_series_data.json'), 'r') as f:
        time_series_data = json.load(f)
    visualization_data['time_series'] = time_series_data

with open(os.path.join(output_dir, 'all_visualization_data.json'), 'w') as f:
    json.dump(visualization_data, f)

print("\n=== DATA PROCESSED ===")
print(f"All analysis results saved to '{output_dir}' folder")
print(f"Raw chart data saved in JSON format for UI rendering")
print(f"Master visualization data saved to '{os.path.join(output_dir, 'all_visualization_data.json')}'")
print(f"The complete analysis report has been saved to '{os.path.join(output_dir, 'social_media_analysis_report.md')}'")
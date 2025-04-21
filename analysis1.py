import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from textblob import TextBlob
import re
from collections import Counter
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

input_folder = "input"
if not os.path.exists(input_folder):
    os.makedirs(input_folder)

def ensure_nltk_resources():

    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in resources:
        try:
            if not os.path.exists(os.path.join(nltk_data_path, resource_path)):
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, download_dir=nltk_data_path, quiet=True)
                
            if resource_name == 'punkt':
                word_tokenize("Test sentence.")
                logger.info("NLTK punkt tokenizer verified.")
            elif resource_name == 'stopwords':
                stopwords.words('english')
                logger.info("NLTK stopwords verified.")
        except Exception as e:
            logger.error(f"Error with NLTK resource {resource_name}: {e}")
            return False
    
    return True

nltk_resources_available = ensure_nltk_resources()

def fallback_tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().split()


# Load the JSON data
def load_data(file_path):
    try:
        logger.info(f"Attempting to load data from {file_path}")
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded data with {len(data)} records")
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
    
    logger.info("Using sample data instead")
    sample_data = [
        {"kind": "t3", "data": {"subreddit": "Anarchism", "author": "AutoModerator", "title": "What Are You Reading/Book Club Tuesday", "created_utc": 1739858460.0, "score": 2, "ups": 2, "downs": 0, "num_comments": 1}},
        {"kind": "t3", "data": {"subreddit": "Anarchism", "author": "NewMunicipalAgenda", "title": "\"WTF is Social Ecology?\" by Usufruct Collective", "created_utc": 1739818025.0, "score": 48, "ups": 48, "downs": 0, "num_comments": 2}}
    ]
    return sample_data

def extract_keywords(text, num_keywords=5):
    """Extract keywords with better error handling and using NLTK resources properly"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    if nltk_resources_available:
        try:
            words = word_tokenize(clean_text(text))
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            word_counts = Counter(filtered_words)
            return word_counts.most_common(num_keywords)
        except Exception as e:
            logger.error(f"Error using NLTK for keyword extraction: {e}")
    
    logger.warning("Using fallback tokenizer for keywords extraction")
    words = fallback_tokenize(clean_text(text))
    
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                 'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'to', 'of',
                 'for', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from',
                 'up', 'down', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'this',
                 'that', 'these', 'those', 'they', 'them', 'their', 'his', 'her', 'its'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(num_keywords)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_subjectivity(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.subjectivity

def categorize_content(title, selftext):
    """Simple content categorization based on keywords"""
    categories = []
    combined_text = (title + " " + selftext).lower() if isinstance(selftext, str) else title.lower()
    
    politics_keywords = ['political', 'government', 'democracy', 'election', 'vote', 'policy', 'freedom', 'rights', 'anarchism']
    
    educational_keywords = ['learn', 'education', 'book', 'reading', 'guide', 'tutorial', 'explain', 'understanding', 'wtf is']
    
    discussion_keywords = ['discussion', 'debate', 'opinion', 'thoughts', 'what are you', 'what do you think', 'question']
    
    if any(keyword in combined_text for keyword in politics_keywords):
        categories.append('Politics')
    
    if any(keyword in combined_text for keyword in educational_keywords):
        categories.append('Educational')
    
    if any(keyword in combined_text for keyword in discussion_keywords):
        categories.append('Discussion')
    
    if not categories:
        categories.append('Other')
    
    return categories

# Function to generate content insights using Gemini API
def generate_content_insights(df, api_key=None):
    """
    Generate advanced insights on Reddit content using the Gemini API.
    
    Args:
        df: Pandas DataFrame containing processed Reddit posts
        api_key: API key for the Gemini service
    
    Returns:
        dict: Dictionary containing different analysis results
    """
    if not api_key:
        logger.info("No API key provided, using basic analysis only.")
        return generate_fallback_analysis(df)
    
    titles = df['title'].tolist()[:20] 
    combined_titles = "\n".join([f"- {title}" for title in titles])
    
    if 'selftext' in df.columns:
        content_samples = []
        for text in df['selftext'].dropna().head(5):
            if isinstance(text, str) and len(text) > 10:
                content_samples.append(text[:500] + "..." if len(text) > 500 else text)
        combined_content = "\n\n".join(content_samples)
    else:
        combined_content = "No content available for analysis."
    
    # Prompts for different analysis types
    analysis_prompts = {
        "thematic": f"Identify 3-5 key themes or topics from these Reddit post titles:\n{combined_titles}",
        "sentiment_context": f"Beyond basic sentiment scores, analyze the emotional tone of these Reddit posts and explain possible context:\n{combined_titles}",
        "user_intent": f"Categorize the apparent user intent behind these Reddit posts (e.g., seeking information, sharing resources, community building):\n{combined_titles}",
        "content_depth": f"Analyze the depth and substance of these sample Reddit posts:\n{combined_content}"
    }
    
    insights = {}
    
    for analysis_type, prompt in analysis_prompts.items():
        try:
            result = call_llm_api(prompt, api_key)
            insights[analysis_type] = result
        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {e}")
            insights[analysis_type] = f"Analysis failed: {str(e)}"
    
    return insights

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm_api(prompt, api_key):
    """
    Call the Gemini API with retry logic
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    # Gemini API URL
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        response_json = response.json()
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            if 'content' in response_json['candidates'][0]:
                content = response_json['candidates'][0]['content']
                if 'parts' in content and len(content['parts']) > 0:
                    return content['parts'][0]['text']
        return "No text content found in response"
    else:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")

def generate_fallback_analysis(df):
    """Generate basic insights without using an LLM API"""
    insights = {}
    
    # Basic thematic analysis using categories
    if 'categories' in df.columns:
        category_counts = df['categories'].str.split(', ').explode().value_counts()
        insights["thematic"] = f"Most common categories in posts: {', '.join(category_counts.index[:3])}"
    else:
        insights["thematic"] = "No category information available."
    
    # Basic sentiment context
    avg_sentiment = df['combined_sentiment'].mean() if 'combined_sentiment' in df.columns else 0
    sentiment_description = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
    insights["sentiment_context"] = f"Posts have an overall {sentiment_description} tone with average sentiment score of {avg_sentiment:.2f}."
    
    # Basic user intent analysis
    has_questions = df['title'].str.contains(r'\?').mean() if 'title' in df.columns else 0
    insights["user_intent"] = f"Approximately {has_questions*100:.1f}% of posts contain questions, suggesting information seeking behavior."
    
    # Basic content depth analysis
    if 'selftext' in df.columns:
        avg_length = df['selftext'].str.len().mean()
        insights["content_depth"] = f"Average post length is {avg_length:.0f} characters, suggesting {'detailed' if avg_length > 500 else 'brief'} content."
    else:
        insights["content_depth"] = "No content text available for analysis."
    
    return insights

def format_llm_insights(insights):
    """Format the LLM insights for inclusion in the final report"""
    if not insights:
        return "No AI-enhanced insights available."
    
    formatted = "## AI-Enhanced Content Insights\n\n"
    
    if "thematic" in insights:
        formatted += "### Key Themes\n" + insights["thematic"] + "\n\n"
    
    if "sentiment_context" in insights:
        formatted += "### Sentiment Context\n" + insights["sentiment_context"] + "\n\n"
    
    if "user_intent" in insights:
        formatted += "### User Intent Analysis\n" + insights["user_intent"] + "\n\n"
    
    if "content_depth" in insights:
        formatted += "### Content Depth\n" + insights["content_depth"] + "\n\n"
    
    return formatted


# Main execution

try:
    data = load_data('input.json')
except:
    logger.warning("Could not load data from input.json, using sample data")
   
posts_data = []
for post in data:
    if 'data' in post:
        post_data = post['data']
        
        created_utc = post_data.get('created_utc', 0)
        created_datetime = datetime.utcfromtimestamp(created_utc) if created_utc else None
        
        title = post_data.get('title', '')
        selftext = post_data.get('selftext', '')
        
        title_sentiment = get_sentiment(title)
        content_sentiment = get_sentiment(selftext)
        combined_sentiment = (title_sentiment + content_sentiment) / 2 if selftext else title_sentiment
        
        title_subjectivity = get_subjectivity(title)
        content_subjectivity = get_subjectivity(selftext)
        combined_subjectivity = (title_subjectivity + content_subjectivity) / 2 if selftext else title_subjectivity
        
        try:
            keywords = extract_keywords(title + " " + selftext if selftext else title)
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            keywords = []
        
        categories = categorize_content(title, selftext)
        
        is_crosspost = 'crosspost_parent' in post_data
        crosspost_parent = post_data.get('crosspost_parent', '')
        
        processed_data = {
            'id': post_data.get('id', ''),
            'subreddit': post_data.get('subreddit', ''),
            'author': post_data.get('author', ''),
            'title': title,
            'selftext': selftext,
            'created_utc': created_datetime,
            'score': post_data.get('score', 0),
            'ups': post_data.get('ups', 0),
            'downs': post_data.get('downs', 0),
            'upvote_ratio': post_data.get('upvote_ratio', 0),
            'num_comments': post_data.get('num_comments', 0),
            'title_sentiment': title_sentiment,
            'content_sentiment': content_sentiment,
            'combined_sentiment': combined_sentiment,
            'title_subjectivity': title_subjectivity,
            'content_subjectivity': content_subjectivity,
            'combined_subjectivity': combined_subjectivity,
            'keywords': ', '.join([kw[0] for kw in keywords]),
            'categories': ', '.join(categories),
            'is_crosspost': is_crosspost,
            'crosspost_parent': crosspost_parent,
            'over_18': post_data.get('over_18', False),
            'stickied': post_data.get('stickied', False),
            'domain': post_data.get('domain', ''),
            'url': post_data.get('url', '')
        }
        
        posts_data.append(processed_data)

df = pd.DataFrame(posts_data)

if 'created_utc' in df.columns and not df['created_utc'].isnull().all():
    df['date'] = df['created_utc'].dt.date
    df['day_of_week'] = df['created_utc'].dt.day_name()
    df['hour_of_day'] = df['created_utc'].dt.hour

try:
    df.to_csv(os.path.join(input_folder, 'processed_posts.csv'), index=False)
    print(f"Processed data saved to {os.path.join(input_folder, 'processed_posts.csv')}")
except Exception as e:
    print(f"Error saving processed data: {e}")

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY environment variable not found.")
    print("AI-enhanced insights will not be available.")
try:
    llm_insights = generate_content_insights(df, api_key=gemini_api_key)
    llm_insights_formatted = format_llm_insights(llm_insights)
    print("Generated AI insights successfully")
except Exception as e:
    print(f"Error generating AI insights: {e}")
    llm_insights_formatted = "Error generating AI insights. Using basic analysis only."

# VISUALIZATIONS AND ANALYSIS

visualization_data = {}

def save_visualization_data(data_key, data_value, description=None):
    """
    Save visualization data to JSON instead of creating PNG files
    
    Args:
        data_key: Key to use in the visualization_data dictionary
        data_value: The data to save (will be converted to serializable format)
        description: Optional description of the data
    """
    try:
        if isinstance(data_value, pd.Series):
            serialized_data = {
                'index': data_value.index.tolist(),
                'values': data_value.values.tolist()
            }
        elif isinstance(data_value, pd.DataFrame):
            serialized_data = data_value.to_dict(orient='records')
        else:
            serialized_data = data_value
            
        visualization_data[data_key] = {
            'data': serialized_data,
            'description': description or f"Data for {data_key}"
        }
        print(f"Saved data for visualization: {data_key}")
    except Exception as e:
        print(f"Error saving visualization data for {data_key}: {e}")

# Time-based analysis
if 'date' in df.columns and not df['date'].isnull().all():
    # Posts per day
    posts_per_day = df.groupby('date').size()
    
    if not posts_per_day.empty:
        posts_per_day_data = {
            'dates': [str(date) for date in posts_per_day.index],
            'counts': posts_per_day.values.tolist()
        }
        save_visualization_data('posts_per_day', posts_per_day_data, 'Number of posts per day')
    
    # Posts by hour of day 
    if 'hour_of_day' in df.columns:
        posts_by_hour = df.groupby('hour_of_day').size()
        
        if not posts_by_hour.empty:
            posts_by_hour_data = {
                'hours': posts_by_hour.index.tolist(),
                'counts': posts_by_hour.values.tolist()
            }
            save_visualization_data('posts_by_hour', posts_by_hour_data, 'Number of posts by hour of day')
            
    # Posts by day of week
    if 'day_of_week' in df.columns:
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        posts_by_day = df.groupby('day_of_week').size()
        
        if not posts_by_day.empty:
            try:
                posts_by_day = posts_by_day.reindex(ordered_days)
            except:
                pass 
            posts_by_day_data = {
                'days': posts_by_day.index.tolist(),
                'counts': posts_by_day.values.tolist()
            }
            save_visualization_data('posts_by_day', posts_by_day_data, 'Number of posts by day of week')

# User analysis
user_stats = df['author'].value_counts()

if not user_stats.empty:
    top_users_data = {
        'users': user_stats.head(min(10, len(user_stats))).index.tolist(),
        'counts': user_stats.head(min(10, len(user_stats))).values.tolist()
    }
    save_visualization_data('top_users', top_users_data, 'Top most active users')

# Subreddit analysis
subreddit_stats = df['subreddit'].value_counts()

if not subreddit_stats.empty:
    top_subreddits_data = {
        'subreddits': subreddit_stats.head(min(10, len(subreddit_stats))).index.tolist(),
        'counts': subreddit_stats.head(min(10, len(subreddit_stats))).values.tolist()
    }
    save_visualization_data('top_subreddits', top_subreddits_data, 'Top subreddits by post count')

# Sentiment analysis data
sentiment_distribution_data = {
    'title_sentiment': df['title_sentiment'].tolist(),
    'content_sentiment': df['content_sentiment'].tolist() if 'content_sentiment' in df.columns else [],
    'combined_sentiment': df['combined_sentiment'].tolist()
}
save_visualization_data('sentiment_distribution', sentiment_distribution_data, 'Distribution of sentiment values')

# Sentiment vs. Engagement Analysis
sentiment_engagement_data = {
    'sentiment_vs_score': df[['combined_sentiment', 'score']].to_dict('records'),
    'sentiment_vs_comments': df[['combined_sentiment', 'num_comments']].to_dict('records'),
    'subjectivity_vs_score': df[['combined_subjectivity', 'score']].to_dict('records'),
    'subjectivity_vs_comments': df[['combined_subjectivity', 'num_comments']].to_dict('records')
}
save_visualization_data('sentiment_engagement', sentiment_engagement_data, 'Relationships between sentiment/subjectivity and engagement metrics')

# Content Category Analysis
if 'categories' in df.columns:
    all_categories = []
    for cats in df['categories'].str.split(', '):
        if isinstance(cats, list):
            all_categories.extend(cats)
    
    category_counts = Counter(all_categories)
    
    if category_counts:
        category_data = {
            'categories': list(category_counts.keys()),
            'counts': list(category_counts.values())
        }
        save_visualization_data('content_categories', category_data, 'Distribution of content categories')

#  Keyword analysis
if 'keywords' in df.columns:
    all_keywords = []
    for kws in df['keywords'].str.split(', '):
        if isinstance(kws, list):
            all_keywords.extend([k for k in kws if k])  
    
    keyword_counts = Counter(all_keywords)
    
    if keyword_counts:
        keyword_data = {
            'keywords': [word for word, _ in keyword_counts.most_common(15)],
            'counts': [count for _, count in keyword_counts.most_common(15)]
        }
        save_visualization_data('top_keywords', keyword_data, 'Top 15 most frequent keywords')

#  Crosspost Analysis
if 'is_crosspost' in df.columns:
    crosspost_count = df['is_crosspost'].sum()
    direct_post_count = len(df) - crosspost_count
    
    if crosspost_count > 0 or direct_post_count > 0:
        crosspost_data = {
            'labels': ['Direct Posts', 'Crossposts'],
            'values': [int(direct_post_count), int(crosspost_count)]
        }
        save_visualization_data('crosspost_ratio', crosspost_data, 'Ratio of direct posts vs. crossposts')
        
        if crosspost_count > 0:
            try:
                crosspost_df = df[df['is_crosspost'] == True].copy()
                if not crosspost_df.empty and 'crosspost_parent_subreddit' in crosspost_df.columns:
                    network_nodes = list(set(crosspost_df['subreddit'].tolist() + crosspost_df['crosspost_parent_subreddit'].tolist()))
                    network_links = []
                    for _, row in crosspost_df.iterrows():
                        if pd.notna(row['crosspost_parent_subreddit']):
                            network_links.append({
                                'source': row['crosspost_parent_subreddit'],
                                'target': row['subreddit'],
                                'value': 1
                            })
                    
                    network_data = {
                        'nodes': [{'id': node, 'group': 1} for node in network_nodes],
                        'links': network_links
                    }
                    save_visualization_data('crosspost_network', network_data, 'Network of crosspost relationships between subreddits')
            except Exception as e:
                print(f"Error creating crosspost network data: {e}")

# Domain Analysis
if 'domain' in df.columns and not df['domain'].isnull().all():
    domain_stats = df['domain'].value_counts()
    
    if not domain_stats.empty:
        domain_data = {
            'domains': domain_stats.head(min(10, len(domain_stats))).index.tolist(),
            'counts': domain_stats.head(min(10, len(domain_stats))).values.tolist()
        }
        save_visualization_data('top_domains', domain_data, 'Top domains by post count')

# Top posts analysis
top_posts = df.nlargest(10, 'score')[['title', 'author', 'subreddit', 'score', 'num_comments']]
top_posts_data = top_posts.to_dict('records')
save_visualization_data('top_posts', top_posts_data, 'Top 10 posts by score')

def try_value(func, default="N/A"):
    try:
        return func()
    except (TypeError, ValueError, KeyError, IndexError):
        return default

# Generate comprehensive analysis report
report = f"""# Reddit Data Analysis Report

## Overview
- **Total Posts Analyzed:** {len(df)}
- **Date Range:** {df['date'].min() if 'date' in df.columns and not df['date'].isnull().all() else 'N/A'} to {df['date'].max() if 'date' in df.columns and not df['date'].isnull().all() else 'N/A'}
- **Total Subreddits:** {df['subreddit'].nunique()}
- **Total Authors:** {df['author'].nunique()}

{llm_insights_formatted}

## Content Analysis
- **Average Sentiment:** {df['combined_sentiment'].mean():.2f} (scale: -1 to 1)
- **Average Subjectivity:** {df['combined_subjectivity'].mean():.2f} (scale: 0 to 1)
- **Most Common Content Categories:** {', '.join([cat for cat, count in Counter(all_categories).most_common(3)]) if 'all_categories' in locals() else 'N/A'}

## Engagement Metrics
- **Average Score:** {df['score'].mean():.2f}
- **Average Comments:** {df['num_comments'].mean():.2f}
- **Average Upvote Ratio:** {f"{df['upvote_ratio'].mean():.2f}" if 'upvote_ratio' in df.columns else 'N/A'}

## Top Subreddits
{subreddit_stats.head(5).to_string()}

## Top Authors
{user_stats.head(5).to_string()}

## Top Posts
{top_posts[['title', 'score', 'author']].head(5).to_string()}

## Sentiment Analysis
- **Most Positive Post:** "{df.loc[df['combined_sentiment'].idxmax(), 'title']}" (Sentiment: {df['combined_sentiment'].max():.2f})
- **Most Negative Post:** "{df.loc[df['combined_sentiment'].idxmin(), 'title']}" (Sentiment: {df['combined_sentiment'].min():.2f})

## Key Findings
1. Content sentiment averages {df['combined_sentiment'].mean():.2f}, indicating an overall {'positive' if df['combined_sentiment'].mean() > 0 else 'negative' if df['combined_sentiment'].mean() < 0 else 'neutral'} tone.
2. {'Posts with higher sentiment scores tend to receive more upvotes.' if df['combined_sentiment'].corr(df['score']) > 0.2 else 'There is no strong correlation between sentiment and post score.'}
3. The most active day for posting is {df['day_of_week'].mode()[0] if 'day_of_week' in df.columns and not df.empty else 'N/A'}.
4. {'Crossposts account for ' + str(round(crosspost_count/len(df)*100, 1)) + '% of all posts.' if 'crosspost_count' in locals() else ''}

## Recommendations
1. Best time to post: {try_value(lambda: df.groupby('hour_of_day').mean()['score'].idxmax(), 'N/A')} UTC on {try_value(lambda: df.groupby('day_of_week').mean()['score'].idxmax(), 'N/A')}
2. Content with {try_value(lambda: 'positive' if df.groupby('combined_sentiment').mean()['score'].idxmax() > 0 else 'negative' if df.groupby('combined_sentiment').mean()['score'].idxmax() < 0 else 'neutral', 'neutral')} sentiment tends to perform better.
3. Most engaging content categories: {try_value(lambda: ', '.join([cat for cat, _ in sorted(zip(df['categories'].unique(), [df[df['categories'] == cat]['score'].mean() for cat in df['categories'].unique()]), key=lambda x: x[1], reverse=True)[:2]]), 'N/A') if 'categories' in df.columns else 'N/A'}
"""

with open(os.path.join(input_folder, 'analysis_report.md'), 'w') as f:
    f.write(report)

with open(os.path.join(input_folder, 'all_visualization_data.json'), 'w') as f:
    json.dump(visualization_data, f, default=str) 

print(f"All analysis results have been saved to the '{input_folder}' folder.")
print(f"Raw data for visualizations saved to '{os.path.join(input_folder, 'all_visualization_data.json')}'")
print("The data can now be used to create visualizations in your frontend.")
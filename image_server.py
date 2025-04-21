from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path
import logging

# Add this logging configuration near the top of your file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JSON_DATA_PATH_INPUT = Path(__file__).parent / "input" / "all_visualization_data.json"
JSON_DATA_PATH_OUTPUT = Path(__file__).parent / "analysis_output" / "all_visualization_data.json"
IMAGE_DIR = Path(__file__).parent / "input"
ANALYSIS_OUTPUT_DIR = Path(__file__).parent / "analysis_output"

def load_data():
    """Load data from JSON file, checking both input and analysis_output directories"""
    if os.path.exists(JSON_DATA_PATH_OUTPUT):
        with open(JSON_DATA_PATH_OUTPUT, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded data from {JSON_DATA_PATH_OUTPUT}")
            return data
    
    if os.path.exists(JSON_DATA_PATH_INPUT):
        with open(JSON_DATA_PATH_INPUT, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded data from {JSON_DATA_PATH_INPUT}")
            return data
    
    raise FileNotFoundError(f"JSON file not found at either {JSON_DATA_PATH_OUTPUT} or {JSON_DATA_PATH_INPUT}")

try:
    visualization_data = load_data()
    print("Data loaded successfully")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Make sure the JSON file exists in either input or analysis_output folders")

@app.get("/api/posts-per-day")
async def get_posts_per_day():
    """Return posts per day data from input directory"""
    try:
        json_path = Path(__file__).parent / "input" / "all_visualization_data.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                posts_data = data.get("posts_per_day")
                if posts_data:
                    return posts_data
    except Exception as e:
        print(f"Error loading posts per day data: {e}")
    
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    return visualization_data.get("posts_per_day")

@app.get("/api/content-categories")
async def get_content_categories():
    """Return content categories data directly from JSON files"""
    for dir_path in ["analysis_output", "input"]:
        try:
            json_path = Path(__file__).parent / dir_path / "all_visualization_data.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    categories_data = data.get("content_categories")
                    if categories_data:
                        return categories_data
        except Exception as e:
            print(f"Error loading content categories from {dir_path}: {e}")
    
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return visualization_data.get("content_categories")

@app.get("/api/top-subreddits")
async def get_top_subreddits():
    """Return top subreddits data directly from JSON files"""
    print("GET /api/top-subreddits endpoint called")
    
    for dir_path in ["analysis_output", "input"]:
        try:
            json_path = Path(__file__).parent / dir_path / "all_visualization_data.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # Try standard key first
                    subreddits_data = data.get("top_subreddits")
                    if subreddits_data:
                        return subreddits_data
                    
                    # Try alternative keys if top_subreddits doesn't exist
                    if dir_path == "input":
                        for alt_key in ["subreddit_distribution", "subreddits", "top_subreddits_data"]:
                            if alt_key in data:
                                return data[alt_key]
        except Exception as e:
            print(f"Error loading top subreddits from {dir_path}: {e}")
    
    # Final fallback - minimal response
    return {"labels": [], "values": [], "error": "No subreddit data found"}


@app.get("/api/sentiment-distribution")
async def get_sentiment_distribution():
    """Return sentiment distribution data"""
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    return visualization_data.get("sentiment_distribution")

@app.get("/api/subreddit-distribution")
async def get_subreddit_distribution():
    """Return subreddit distribution data"""
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    return visualization_data.get("subreddit_distribution") 

@app.get("/api/top-bad-words")
async def get_top_bad_words():
    """Return top bad words data directly from analysis_output folder"""
    try:
        json_path = Path(__file__).parent / "analysis_output" / "all_visualization_data.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                bad_words_data = data.get("top_bad_words")
                if bad_words_data:
                    return bad_words_data
    except Exception as e:
        print(f"Error loading bad words directly: {e}")
    
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    return visualization_data.get("top_bad_words")

@app.get("/api/top-political-words")
async def get_top_political_words():
    """Return top political words data directly from analysis_output folder"""
    try:
        json_path = Path(__file__).parent / "analysis_output" / "all_visualization_data.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                political_words_data = data.get("top_political_words")
                if political_words_data:
                    return political_words_data
    except Exception as e:
        print(f"Error loading political words directly: {e}")
    
    if 'visualization_data' not in globals():
        raise HTTPException(status_code=500, detail="Data not loaded")
    return visualization_data.get("top_political_words")

# Mount static directories
try:
    app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
    app.mount("/output", StaticFiles(directory=str(ANALYSIS_OUTPUT_DIR)), name="output")
except Exception as e:
    print(f"Failed to mount directories: {e}")

@app.get("/api/political-wordcloud")
async def get_political_wordcloud():
    """Serve political wordcloud image"""
    for dir_path, dir_var in [("input", IMAGE_DIR), ("analysis_output", ANALYSIS_OUTPUT_DIR)]:
        image_path = dir_var / "political_words_wordcloud.png"
        if os.path.exists(image_path):
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Political wordcloud image not found")

@app.get("/api/bad-words-wordcloud")
async def get_bad_words_wordcloud():
    """Serve bad words wordcloud image"""
    for dir_path, dir_var in [("input", IMAGE_DIR), ("analysis_output", ANALYSIS_OUTPUT_DIR)]:
        image_path = dir_var / "bad_words_wordcloud.png"
        if os.path.exists(image_path):
            return FileResponse(image_path)
    
    raise HTTPException(status_code=404, detail="Bad words wordcloud image not found")

@app.get("/api/markdown/sentiment-analysis")
async def get_sentiment_analysis():
    """Return sentiment analysis markdown content"""
    try:
        file_path = Path(__file__).parent / "analysis_output" / "social_media_analysis_report.md"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "## Sentiment Analysis" in content:
                section = content.split("## Sentiment Analysis")[1].split("##")[0].strip()
                return {"content": section}
    except Exception as e:
        print(f"Error loading sentiment analysis content: {e}")
    
    # Simple fallback content
    #return {"content": "Posts have an overall positive tone with average sentiment score of 0.42."}

@app.get("/api/markdown/network-analysis")
async def get_network_analysis():
    """Return network analysis markdown content"""
    try:
        file_path = Path(__file__).parent / "analysis_output" / "social_media_analysis_report.md"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "## Network Analysis" in content:
                section = content.split("## Network Analysis")[1].split("##")[0].strip()
                return {"content": section}
    except Exception as e:
        print(f"Error loading network analysis content: {e}")
    
# Add this new endpoint to serve the GraphML data

@app.get("/api/crosspost-network-graphml")
async def get_crosspost_network_graphml():
    """Return Reddit crosspost network as GraphML for visualization"""
    try:
        graphml_path = Path(__file__).parent / "reddit_crosspost_network.graphml"
        logger.info(f"Loading GraphML data from: {graphml_path}")
        
        if os.path.exists(graphml_path):
            with open(graphml_path, 'r', encoding='utf-8') as f:
                graphml_content = f.read()
            return {"graphml": graphml_content}
        else:
            logger.error(f"GraphML file not found: {graphml_path}")
            raise HTTPException(status_code=404, detail="Crosspost network GraphML file not found")
    except Exception as e:
        logger.error(f"Error loading GraphML data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading GraphML data: {str(e)}")
    
@app.get("/api/crosspost-stats")
async def get_crosspost_stats():
    try:
        json_path = Path(__file__).parent / "reddit_crosspost_network.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats = data.get("stats", {})
                
                # Return properly formatted stats data
                return {
                    "subredditCount": len(data.get("nodes", [])),
                    "crosspostCount": stats.get("total_crossposts", 0),
                    "connectionCount": stats.get("unique_connections", 0),
                    "userCount": len(set([crosspost.get("original", {}).get("author", "") 
                                       for crosspost in data.get("crossposts", [])]))
                }
        else:
            # Return mock data when file not found (for development purposes)
            print("Warning: reddit_crosspost_network.json not found, returning mock data")
            return {
                "subredditCount": 42,
                "crosspostCount": 237,
                "connectionCount": 128,
                "userCount": 95
            }
    except Exception as e:
        print(f"Error in /api/crosspost-stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading crosspost stats: {str(e)}")
    
@app.get("/api/crosspost-top")
async def get_crosspost_top():
    """Return top source and destination subreddits from the JSON file"""
    try:
        json_path = Path(__file__).parent / "reddit_crosspost_network.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Get top sources and destinations from the stats
                top_sources = data.get("stats", {}).get("top_source_subreddits", [])
                top_destinations = data.get("stats", {}).get("top_destination_subreddits", [])
                
                return {
                    "sources": [{"name": s.get("name"), "count": s.get("count")} for s in top_sources],
                    "destinations": [{"name": s.get("name"), "count": s.get("count")} for s in top_destinations]
                }
        else:
            raise HTTPException(status_code=404, detail="Crosspost network data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading top subreddits: {str(e)}")
    
@app.get("/api/debug")
async def debug_info():
    """Return debug information about the server"""
    input_image_files = list(IMAGE_DIR.glob("*.png"))
    output_image_files = list(ANALYSIS_OUTPUT_DIR.glob("*.png"))
    
    return {
        "status": "running",
        "input_dir": str(IMAGE_DIR),
        "input_files": [str(f.name) for f in input_image_files],
        "output_dir": str(ANALYSIS_OUTPUT_DIR),
        "output_files": [str(f.name) for f in output_image_files],
        "json_in_input": os.path.exists(JSON_DATA_PATH_INPUT),
        "json_in_output": os.path.exists(JSON_DATA_PATH_OUTPUT),
        "data_loaded": 'visualization_data' in globals()
    }

@app.get("/api/reload-data")
async def reload_data():
    """Force reload data from JSON file"""
    global visualization_data
    try:
        visualization_data = load_data()
        return {"status": "Data reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")

@app.get("/")
async def root():
    data_source = str(JSON_DATA_PATH_OUTPUT if os.path.exists(JSON_DATA_PATH_OUTPUT) else JSON_DATA_PATH_INPUT)
    return {
        "status": "API is running", 
        "data_source": data_source,
        "available_endpoints": [
            "/api/posts-per-day",
            "/api/content-categories",
            "/api/top-subreddits",
            "/api/sentiment-vs-score",
            "/api/sentiment-distribution",
            "/api/subreddit-distribution",
            "/api/top-bad-words",
            "/api/top-political-words"
        ]
    }

@app.get("/api/markdown/{section}")
async def get_markdown_content(section: str):
    """Serve markdown content from files"""
    # Map endpoint names to actual files and sections
    section_mapping = {
        "overview": {"file": "analysis_report.md", "section": "Overview"},
        "content-analysis": {"file": "social_media_analysis_report.md", "section": "Content Analysis"},
        "sentiment-analysis": {"file": "social_media_analysis_report.md", "section": "Sentiment Analysis"},
        "engagement-analysis": {"file": "social_media_analysis_report.md", "section": "Engagement Metrics"},
        "recommendations": {"file": "social_media_analysis_report.md", "section": "Key Findings"},
        "community": {"file": "social_media_analysis_report.md", "section": "Overall Summary"},
        "political-analysis": {"file": "social_media_analysis_report.md", "section": "Political Words and Engagement Analysis"},
        "bad-words": {"file": "social_media_analysis_report.md", "section": "Bad Words Analysis"}
    }
    
    if section not in section_mapping:
        raise HTTPException(status_code=404, detail=f"Markdown section '{section}' not found")
    
    file_info = section_mapping[section]
    md_file = file_info["file"]
    section_heading = file_info["section"]
    
    # Try both directories
    for directory in ["input", "analysis_output"]:
        file_path = Path(__file__).parent / directory / md_file
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Return whole file for overview
                if section == "overview":
                    return {"content": content}
                
                # Extract specific section
                section_pattern = f"## {section_heading}"
                if section_pattern in content:
                    section_start = content.find(section_pattern)
                    section_content = content[section_start:]
                    
                    next_section = section_content.find("\n## ", 5)
                    if next_section > 0:
                        section_content = section_content[:next_section]
                    
                    return {"content": section_content}
                
                # Default to whole content if section not found
                return {"content": content}
            except Exception as e:
                print(f"Error reading markdown file: {e}")
    
    raise HTTPException(status_code=404, detail=f"Markdown file not found: {md_file}")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting data server on http://localhost:8000")
    print(f"JSON data path: {JSON_DATA_PATH_OUTPUT if os.path.exists(JSON_DATA_PATH_OUTPUT) else JSON_DATA_PATH_INPUT}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
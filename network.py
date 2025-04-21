import json
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
from tqdm import tqdm
import time
import argparse
import os
from dotenv import load_dotenv

def create_reddit_crosspost_graph(input_file, neo4j_uri=None, 
                                neo4j_user=None, neo4j_password=None,
                                top_posts_limit=None): 
    """
    Create a Neo4j graph database of Reddit posts and their crossposting relationships
    Creates nodes for ALL crossposts to show the complete network
    """
    start_time = time.time()
    
    load_dotenv()
    
    neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        raise ValueError("Neo4j password not provided. Set NEO4J_PASSWORD environment variable or pass as parameter.")
    
    print("Connecting to Neo4j...")
    graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        posts = json.load(f)
    
    print("Clearing existing data...")
    graph.run("MATCH (n) DETACH DELETE n")
    
    print("Creating indices...")
    graph.run("CREATE INDEX post_id_index IF NOT EXISTS FOR (p:Post) ON (p.id)")
    graph.run("CREATE INDEX subreddit_name_index IF NOT EXISTS FOR (s:Subreddit) ON (s.name)")
    graph.run("CREATE INDEX crosspost_id_index IF NOT EXISTS FOR (c:Crosspost) ON (c.id)")
    
    # Identifying all crosspost relationships
    print("Scanning for crossposts...")
    crosspost_info = []
    for post in tqdm(posts, desc="Scanning posts"):
        post_data = post["data"]
        try:
            if "crosspost_parent" in post_data and post_data["crosspost_parent"]:
                post_id = post_data["id"]
                parent_id = post_data["crosspost_parent"].split("_")[1]
                
                # Find the parent post
                parent_data = None
                for p in posts:
                    if p["data"]["id"] == parent_id:
                        parent_data = p["data"]
                        break
                
                if parent_data is None and "crosspost_parent_list" in post_data:
                    if post_data["crosspost_parent_list"] and len(post_data["crosspost_parent_list"]) > 0:
                        parent_data = post_data["crosspost_parent_list"][0]
                
                if parent_data is not None:
                    source_subreddit = parent_data["subreddit"]
                    dest_subreddit = post_data["subreddit"]
                    
                    crosspost_info.append({
                        "post_id": post_id,
                        "parent_id": parent_id,
                        "source_subreddit": source_subreddit,
                        "dest_subreddit": dest_subreddit,
                        "post_data": post_data,
                        "parent_data": parent_data
                    })
        except (KeyError, IndexError) as e:
            print(f"Error processing post {post_data.get('id', 'unknown')}: {str(e)}")
            continue
    
    print(f"Found {len(crosspost_info)} crossposts")
    
    if len(crosspost_info) == 0:
        print("WARNING: No crossposts found in data. Creating some sample relationships for visualization.")
        subreddits_list = set()
        
        for post in posts[:100]:  
            subreddits_list.add(post["data"]["subreddit"])
        
        subreddits_list = list(subreddits_list)[:10]  
        
        if len(subreddits_list) >= 2:
            import random
            for i in range(min(20, len(subreddits_list) * 2)):
                source = random.choice(subreddits_list)
                dest = random.choice(subreddits_list)
                if source != dest:  
                    weight = random.randint(1, 5)
                    crosspost_info.append({
                        "post_id": f"sample_{i}",
                        "parent_id": f"parent_{i}",
                        "source_subreddit": source,
                        "dest_subreddit": dest,
                        "post_data": {"title": "Sample Post", "author": "sample_user"},
                        "parent_data": {"title": "Sample Parent", "author": "sample_user"}
                    })
                    
            print(f"Created {len(crosspost_info)} sample crosspost relationships for visualization")
    
    subreddit_crosspost_count = {}
    for info in crosspost_info:
        pair = (info["source_subreddit"], info["dest_subreddit"])
        if pair in subreddit_crosspost_count:
            subreddit_crosspost_count[pair] += 1
        else:
            subreddit_crosspost_count[pair] = 1
    
    sorted_pairs = sorted(subreddit_crosspost_count.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(sorted_pairs)} subreddit crosspost relationships")
    
    # Start transaction
    tx = graph.begin()
    
    # Create subreddit nodes for all relevant subreddits
    all_subreddits = set()
    for info in crosspost_info:
        all_subreddits.add(info["source_subreddit"])
        all_subreddits.add(info["dest_subreddit"])
    
    print(f"Creating {len(all_subreddits)} subreddit nodes...")
    subreddits = {} 
    for subreddit_name in all_subreddits:
        subscribers = 0
        for post in posts:
            if post["data"]["subreddit"] == subreddit_name:
                subscribers = post["data"].get("subreddit_subscribers", 0)
                break
        
        subreddit_node = Node("Subreddit", 
                            name=subreddit_name,
                            subscribers=subscribers)
        tx.merge(subreddit_node, "Subreddit", "name")
        subreddits[subreddit_name] = subreddit_node
    
    # Create nodes for ALL crossposts and their relationships
    print(f"Creating nodes for all {len(crosspost_info)} crosspost relationships...")
    for i, info in enumerate(tqdm(crosspost_info, desc="Creating crosspost nodes")):
        post_id = info["post_id"]
        parent_id = info["parent_id"]
        source_subreddit = info["source_subreddit"]
        dest_subreddit = info["dest_subreddit"]
        post_data = info["post_data"]
        parent_data = info["parent_data"]
        
        crosspost_node = Node("Crosspost",
                             id=f"{parent_id}_{post_id}",
                             title=post_data.get("title", ""),
                             parent_title=parent_data.get("title", ""),
                             author=post_data.get("author", "[deleted]"),
                             parent_author=parent_data.get("author", "[deleted]"),
                             score=post_data.get("score", 0),
                             parent_score=parent_data.get("score", 0),
                             created_utc=post_data.get("created_utc", 0))
        tx.create(crosspost_node)
        
        source_node = subreddits[source_subreddit]
        dest_node = subreddits[dest_subreddit]
        
        from_rel = Relationship(crosspost_node, "FROM_SUBREDDIT", source_node)
        to_rel = Relationship(crosspost_node, "TO_SUBREDDIT", dest_node)
        tx.create(from_rel)
        tx.create(to_rel)
        
        # Create parent and child post nodes
        parent_node = Node("Post", 
                        id=parent_id, 
                        title=parent_data.get("title", ""),
                        author=parent_data.get("author", "[deleted]"),
                        score=parent_data.get("score", 0))
        tx.merge(parent_node, "Post", "id")
        
        post_node = Node("Post", 
                       id=post_id, 
                       title=post_data.get("title", ""),
                       author=post_data.get("author", "[deleted]"),
                       score=post_data.get("score", 0))
        tx.merge(post_node, "Post", "id")
        
        # Link posts
        source_posted_rel = Relationship(parent_node, "POSTED_IN", source_node)
        dest_posted_rel = Relationship(post_node, "POSTED_IN", dest_node)
        tx.create(source_posted_rel)
        tx.create(dest_posted_rel)
        
        orig_rel = Relationship(crosspost_node, "ORIGINAL_POST", parent_node)
        repost_rel = Relationship(crosspost_node, "REPOSTED_AS", post_node)
        tx.create(orig_rel)
        tx.create(repost_rel)
        
        crosspost_rel = Relationship(post_node, "CROSSPOST_OF", parent_node)
        tx.create(crosspost_rel)
    
    print("Creating aggregated crosspost relationships between subreddits...")
    for (source, dest), count in sorted_pairs:
        if source != dest: 
            source_node = subreddits[source]
            dest_node = subreddits[dest]
            
            rel = Relationship(dest_node, "CROSSPOST_FROM", source_node, weight=count)
            tx.create(rel)
    
    graph.commit(tx)
    
    edge_count = graph.run("MATCH ()-[r:CROSSPOST_FROM]->() RETURN COUNT(r) AS count").data()[0]["count"]
    print(f"Created {edge_count} aggregated subreddit relationships in the database")
    
    crosspost_count = graph.run("MATCH (c:Crosspost) RETURN COUNT(c) AS count").data()[0]["count"]
    print(f"Created {crosspost_count} individual crosspost nodes in the database")
    
    print(f"Graph creation complete!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return graph

def analyze_crosspost_network(graph):
    """
    Run subreddit-focused analysis on the crosspost network
    """
    print("\n=== SUBREDDIT CROSSPOST ANALYSIS ===")
    
    # Overall top crossposted posts regardless of subreddit
    print("\nTop posts by crosspost count (across all subreddits):")
    result = graph.run("""
        MATCH (p:Post)<-[r:CROSSPOST_OF]-() 
        WITH p, COUNT(r) AS crosspost_count
        ORDER BY crosspost_count DESC
        LIMIT 10
        RETURN p.id AS post_id, p.title AS title, p.author AS author, 
               p.score AS score, crosspost_count,
               [(p)-[:POSTED_IN]->(s) | s.name][0] AS original_subreddit,
               [(p)<-[:CROSSPOST_OF]-(repost)-[:POSTED_IN]->(dest) | dest.name] AS crossposted_to
    """)
    
    for record in result:
        destinations = list(set(record["crossposted_to"])) 
        top_destinations = destinations[:5] if len(destinations) > 5 else destinations
        more_count = len(destinations) - len(top_destinations)
        
        print(f"\n  Post: {record['post_id']} - {record['crosspost_count']} crossposts")
        print(f"  From: r/{record['original_subreddit']}")
        print(f"  Title: '{record['title']}'")
        print(f"  Author: {record['author']} (Score: {record['score']})")
        print(f"  Crossposted to: r/{', r/'.join(top_destinations)}" + 
              (f" and {more_count} more subreddits" if more_count > 0 else ""))
    
    # Subreddits with the most crossposted content
    print("\nSubreddits with most crossposted content (sources):")
    result = graph.run("""
        MATCH (source:Subreddit)<-[:FROM_SUBREDDIT]-(c:Crosspost)
        WITH source, COUNT(c) AS crosspost_count
        ORDER BY crosspost_count DESC
        LIMIT 10
        RETURN source.name AS subreddit, crosspost_count
    """)
    
    for record in result:
        print(f"  r/{record['subreddit']}: {record['crosspost_count']} crossposts from this subreddit")
    
    #  Most active crosspost paths between subreddits
    print("\nMost active crosspost paths between subreddits:")
    result = graph.run("""
        MATCH (source:Subreddit)<-[:FROM_SUBREDDIT]-(c:Crosspost)-[:TO_SUBREDDIT]->(dest:Subreddit)
        WHERE source.name <> dest.name
        WITH source.name AS source_subreddit, dest.name AS dest_subreddit, COUNT(c) AS crosspost_count
        ORDER BY crosspost_count DESC
        LIMIT 15
        RETURN source_subreddit, dest_subreddit, crosspost_count
    """)
    
    for record in result:
        print(f"  r/{record['source_subreddit']} → r/{record['dest_subreddit']}: {record['crosspost_count']} crossposts")
    
    #  Individual examples of crossposts
    print("\nExamples of individual crossposts:")
    result = graph.run("""
        MATCH (c:Crosspost)
        MATCH (c)-[:FROM_SUBREDDIT]->(source:Subreddit)
        MATCH (c)-[:TO_SUBREDDIT]->(dest:Subreddit)
        WHERE source.name <> dest.name
        RETURN c.title AS title, c.author AS author, source.name AS from_subreddit, dest.name AS to_subreddit
        LIMIT 10
    """)
    
    for record in result:
        print(f"  '{record['title']}' by u/{record['author']}")
        print(f"    From r/{record['from_subreddit']} to r/{record['to_subreddit']}")
    
    # Summary stats
    result = graph.run("""
        MATCH (s:Subreddit)
        RETURN COUNT(s) AS subreddit_count
    """).data()[0]
    print(f"\nSummary statistics:")
    print(f"  Total subreddits: {result['subreddit_count']}")
    
    result = graph.run("""
        MATCH (c:Crosspost)
        RETURN COUNT(c) AS crosspost_count
    """).data()[0]
    print(f"  Total crosspost relationships: {result['crosspost_count']}")
    
    result = graph.run("""
        MATCH (source:Subreddit)<-[:FROM_SUBREDDIT]-(c:Crosspost)-[:TO_SUBREDDIT]->(dest:Subreddit)
        WHERE source.name <> dest.name
        RETURN COUNT(DISTINCT [source.name, dest.name]) AS subreddit_connections
    """).data()[0]
    print(f"  Unique subreddit connections: {result['subreddit_connections']}")

def export_for_visualization(graph, output_file="reddit_crosspost_network.graphml"):
    """
    Export the full crosspost network to GraphML format for visualization in tools like Gephi
    Includes all individual crosspost nodes connecting source and destination subreddits
    """
    print(f"\nExporting complete crosspost network for visualization...")
    
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    subreddit_result = graph.run("""
        MATCH (s:Subreddit)
        OPTIONAL MATCH (s)<-[:FROM_SUBREDDIT]-(c1:Crosspost)
        WITH s, COUNT(c1) AS source_count
        OPTIONAL MATCH (s)<-[:TO_SUBREDDIT]-(c2:Crosspost)
        WITH s, source_count, COUNT(c2) AS dest_count
        RETURN s.name AS name, source_count, dest_count, s.subscribers AS subscribers
    """)
    
    for record in subreddit_result:
        name = record["name"]
        source_count = record["source_count"]
        dest_count = record["dest_count"]
        
        G.add_node(
            f"sub_{name}", 
            type="subreddit",
            label=f"r/{name}",
            source_count=source_count,
            dest_count=dest_count,
            subscribers=record["subscribers"] or 0,
            size=10 + source_count + dest_count 
        )
    
    crosspost_result = graph.run("""
        MATCH (c:Crosspost)
        MATCH (c)-[:FROM_SUBREDDIT]->(source:Subreddit)
        MATCH (c)-[:TO_SUBREDDIT]->(dest:Subreddit)
        WHERE source.name <> dest.name
        RETURN c.id AS id, c.title AS title, c.author AS author, 
               source.name AS source_subreddit, dest.name AS dest_subreddit
    """)
    
    for record in crosspost_result:
        crosspost_id = record["id"]
        source = record["source_subreddit"]
        dest = record["dest_subreddit"]
        
        G.add_node(
            f"cp_{crosspost_id}", 
            type="crosspost",
            label=record["title"][:20] + "..." if len(record["title"]) > 20 else record["title"],
            title=record["title"],
            author=record["author"],
            size=5  
        )
        
        G.add_edge(f"cp_{crosspost_id}", f"sub_{source}", type="from_subreddit")
        G.add_edge(f"cp_{crosspost_id}", f"sub_{dest}", type="to_subreddit")
    
    # Write to GraphML format
    nx.write_graphml(G, output_file)
    
    crosspost_count = len([n for n in G.nodes() if G.nodes[n]["type"] == "crosspost"])
    subreddit_count = len([n for n in G.nodes() if G.nodes[n]["type"] == "subreddit"])
    
    print(f"Exported network with {subreddit_count} subreddits and {crosspost_count} crosspost nodes")
    print(f"Network saved to {output_file} for visualization in Gephi or other tools")
    
    return G

def visualize_network(G, output_file="reddit_crosspost_network.png"):
    """
    Create a visualization of the complete crosspost network
    Shows subreddits and all individual crosspost nodes
    """
    print(f"\nCreating visualization of all crosspost relationships...")
    
    plt.figure(figsize=(24, 18), dpi=300)
    
    subreddit_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "subreddit"]
    crosspost_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "crosspost"]
    
    print(f"Visualizing {len(subreddit_nodes)} subreddits and {len(crosspost_nodes)} crossposts")
    
    print("Calculating network layout (this may take a while)...")
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    from_edges = [(u, v) for u, v in G.edges() if "from_subreddit" in G.edges[u, v].get("type", "")]
    to_edges = [(u, v) for u, v in G.edges() if "to_subreddit" in G.edges[u, v].get("type", "")]
    
    nx.draw_networkx_edges(G, pos, edgelist=from_edges, alpha=0.4, edge_color="green", arrows=True, width=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=to_edges, alpha=0.4, edge_color="red", arrows=True, width=0.8)
    
    subreddit_sizes = [G.nodes[n].get("size", 10) * 10 for n in subreddit_nodes] 
    nx.draw_networkx_nodes(G, pos, nodelist=subreddit_nodes, node_size=subreddit_sizes, 
                           node_color="skyblue", alpha=0.8, edgecolors="black", linewidths=1)
    
    nx.draw_networkx_nodes(G, pos, nodelist=crosspost_nodes, node_size=30, 
                           node_color="orange", alpha=0.6)
    
    labels = {n: G.nodes[n].get("label", n) for n in subreddit_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
    
    plt.plot([0], [0], 'o', color='skyblue', markersize=10, label='Subreddit')
    plt.plot([0], [0], 'o', color='orange', markersize=5, label='Crosspost')
    plt.plot([0], [0], '-', color='green', linewidth=2, label='From Subreddit')
    plt.plot([0], [0], '-', color='red', linewidth=2, label='To Subreddit')
    plt.legend(loc='upper left', fontsize=12)
    
    subreddit_count = len(subreddit_nodes)
    crosspost_count = len(crosspost_nodes)
    
    plt.suptitle(f"Reddit Crosspost Network: {crosspost_count} Crossposts Between {subreddit_count} Subreddits", 
                 fontsize=24, y=0.98)
    
    plt.figtext(0.5, 0.01, 
                "Each orange node represents an individual crosspost connecting two subreddits.\n" +
                "Green lines show the original subreddit, red lines show the destination subreddit.", 
                ha='center', fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    svg_file = os.path.splitext(output_file)[0] + ".svg"
    plt.savefig(svg_file, format="svg", bbox_inches="tight")
    
    print(f"Visualization saved to {output_file} and {svg_file}")
    
    return plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit Crosspost Network Analysis - All Crossposts")
    parser.add_argument("--input", default="input.json", help="Input JSON file")
    parser.add_argument("--neo4j-uri", default="neo4j+s://8bc46ceb.databases.neo4j.io", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="RximALodxdQCU-1wGEd8KAQvSTkRnoFuiadWb86ElxY", help="Neo4j password")
    parser.add_argument("--skip-import", action="store_true", help="Skip data import and use existing database")
    parser.add_argument("--analysis-only", action="store_true", help="Only run analysis on existing database")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    
    args = parser.parse_args()
    start_time = time.time()
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    graphml_path = os.path.join(output_dir, "reddit_crosspost_network.graphml")
    png_path = os.path.join(output_dir, "reddit_crosspost_network.png")
    json_path = os.path.join(output_dir, "reddit_crosspost_network.json")
    
    # Connect to Neo4j
    if args.analysis_only:
        print("Connecting to Neo4j for analysis only...")
        graph = Graph(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    elif args.skip_import:
        print("Connecting to existing database...")
        graph = Graph(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    else:
        graph = create_reddit_crosspost_graph(
            args.input,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user, 
            neo4j_password=args.neo4j_password,
            top_posts_limit=None 
        )
    
    analyze_crosspost_network(graph)
    
    G = export_for_visualization(graph, output_file=graphml_path)
    
    plt = visualize_network(G, output_file=png_path)
    
 
    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds")
    print(f"\nOutputs:")
    print(f"- Network Visualization: {png_path}")
    print(f"- GraphML Network File: {graphml_path}")

import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml("../kongyiji/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(notebook=True, cdn_resources='remote')

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Add colors and title to nodes
for node in net.nodes:
    node["title"] = "\n\n".join([f"entity_id : {node['entity_id']}", f"entity_type : {node['entity_type']}",
                                 f"description : {node['description']}", f"source_id : {node['source_id']}",
                                 f"file_path : {node['file_path']}"])
    # ["organization", "person", "geo", "event", "category"]
    if node["entity_type"] == "organization":
        node["color"] = "#FFA500"
    elif node["entity_type"] == "person":
        node["color"] = "#00FF00"
    elif node["entity_type"] == "geo":
        node["color"] = "#0000FF"
    elif node["entity_type"] == "event":
        node["color"] = "#FF0000"
    elif node["entity_type"] == "category":
        node["color"] = "#A5FF00"
    else:
        node["color"] = "#000000"

# Add title to edges
for edge in net.edges:
    edge["title"] = "\n\n".join([f"from : {edge['from']}", f"to : {edge['to']}", f"description : {edge['description']}",
                                 f"keywords : {edge['keywords']}", f"width : {edge['width']}",
                                 f"source_id : {edge['source_id']}", f"file_path : {edge['file_path']}"])
    if edge["width"] > 10:
        edge["color"] = "#FF0000"
    elif edge["width"] > 5:
        edge["color"] = "#00FF00"
    else:
        edge["color"] = "#0000FF"

# Save and display the network
net.show("knowledge_graph.html")

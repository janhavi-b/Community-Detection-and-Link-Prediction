
import networkx as nx
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "facebook_combined.txt"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load the undirected social network
G = nx.read_edgelist(
    DATA_FILE,
    nodetype=int,
    create_using=nx.Graph()
)

# Remove self-loops, if any
G.remove_edges_from(nx.selfloop_edges(G))

print("DATASET LOADED SUCCESSFULLY")
print("-" * 35)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Is graph directed:", G.is_directed())
print("Number of connected components:",
      nx.number_connected_components(G))
print("Average degree:", round(
    sum(dict(G.degree()).values()) / G.number_of_nodes(), 2
))

# Save basic network statistics
stats = pd.DataFrame({
    "Metric": [
        "Number of nodes",
        "Number of edges",
        "Average degree",
        "Connected components"
    ],
    "Value": [
        G.number_of_nodes(),
        G.number_of_edges(),
        round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        nx.number_connected_components(G)
    ]
})

stats.to_csv(RESULTS_DIR / "network_statistics.csv", index=False)
print("\nStatistics saved to results/network_statistics.csv")

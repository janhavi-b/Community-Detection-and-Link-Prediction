
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import community as community_louvain
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "facebook_combined.txt"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Load network
G = nx.read_edgelist(
    DATA_FILE,
    nodetype=int,
    create_using=nx.Graph()
)
G.remove_edges_from(nx.selfloop_edges(G))

print("Detecting communities using Louvain...")

# Method 1: Disjoint community detection
partition = community_louvain.best_partition(
    G, random_state=42
)

# Number of communities
community_ids = set(partition.values())
print("Total communities detected:", len(community_ids))

# Community sizes
community_sizes = pd.Series(partition).value_counts().sort_index()
print("\nCommunity sizes:")
print(community_sizes.to_string())

# Evaluate using modularity
modularity = community_louvain.modularity(partition, G)
print("\nLouvain modularity:", round(modularity, 4))

# Save node-to-community assignments
community_df = pd.DataFrame({
    "Node": list(partition.keys()),
    "Community": list(partition.values())
})
community_df.to_csv(
    TABLES_DIR / "louvain_communities.csv", index=False
)

# Save community size table
community_sizes.rename("Number_of_Nodes").to_csv(
    TABLES_DIR / "louvain_community_sizes.csv"
)

# Visualize network
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, iterations=30)

node_colors = [partition[node] for node in G.nodes()]

nx.draw_networkx_edges(
    G, pos, alpha=0.12, width=0.4, edge_color="gray"
)
nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    node_size=12,
    linewidths=0
)

plt.title(
    f"Louvain Community Detection\n"
    f"Communities: {len(community_ids)} | "
    f"Modularity: {modularity:.4f}"
)
plt.axis("off")
plt.tight_layout()

output_file = FIGURES_DIR / "louvain_communities.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close()

print("\nGraph saved to:", output_file)
print("Community assignments saved successfully.")

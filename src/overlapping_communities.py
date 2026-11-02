import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random

# Project paths
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "facebook_combined.txt"
FIG = BASE / "results" / "figures"
TAB = BASE / "results" / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Load dataset
print("Loading network...", flush=True)
G = nx.read_edgelist(DATA, nodetype=int)

# Select reproducible sample of 150 nodes
random.seed(42)
nodes = random.sample(list(G.nodes()), min(150, len(G)))
H = G.subgraph(nodes).copy()

print("Sample nodes:", H.number_of_nodes(), flush=True)
print("Sample edges:", H.number_of_edges(), flush=True)

# Clique Percolation Method
print("Running CPM with k=4...", flush=True)
communities = list(nx.community.k_clique_communities(H, 4))

print("Communities detected:", len(communities), flush=True)

# Save community memberships
rows = []

for cid, comm in enumerate(communities):
    for node in comm:
        rows.append([node, cid])

df = pd.DataFrame(rows, columns=["Node", "Community"])
df.to_csv(TAB / "overlapping_communities.csv", index=False)

# Identify overlapping nodes
if not df.empty:
    counts = df.groupby("Node")["Community"].nunique()
    overlapping = counts[counts > 1]
else:
    overlapping = pd.Series(dtype=int)

print("Overlapping nodes:", len(overlapping), flush=True)

# Save community sizes
sizes = pd.DataFrame({
    "Community": range(len(communities)),
    "Size": [len(c) for c in communities]
})

sizes.to_csv(TAB / "overlapping_community_sizes.csv", index=False)

# Visualize communities
print("Creating graph...", flush=True)

pos = nx.spring_layout(H, seed=42, iterations=10)

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(H, pos, alpha=0.15)

covered = set()

for cid, comm in enumerate(communities):
    covered.update(comm)

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=list(comm),
        node_color=[plt.cm.tab20(cid % 20)],
        node_size=30
    )

uncovered = set(H.nodes()) - covered

nx.draw_networkx_nodes(
    H,
    pos,
    nodelist=list(uncovered),
    node_color="lightgray",
    node_size=12
)

plt.title("Overlapping Communities - CPM (k=4, 150-node sample)")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    FIG / "overlapping_communities.png",
    dpi=150
)

plt.close()

print("\nSUCCESS: CPM completed!")
print("Community count:", len(communities))
print("Overlapping nodes:", len(overlapping))
print("Files saved in results folder.")
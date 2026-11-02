
import networkx as nx
import community as community_louvain
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Project paths
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "facebook_combined.txt"
TAB = BASE / "results" / "tables"
FIG = BASE / "results" / "figures"

TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Load full network and create the same 150-node sample
print("Loading network...", flush=True)
G = nx.read_edgelist(DATA, nodetype=int)

random.seed(42)
nodes = random.sample(list(G.nodes()), min(150, len(G)))
H = G.subgraph(nodes).copy()

print("Sample nodes:", H.number_of_nodes())
print("Sample edges:", H.number_of_edges())

# Method 1: Louvain disjoint communities
print("\nRunning Louvain...", flush=True)

louvain_partition = community_louvain.best_partition(
    H, random_state=42
)

louvain_communities = {}
for node, cid in louvain_partition.items():
    louvain_communities.setdefault(cid, set()).add(node)

louvain_modularity = community_louvain.modularity(
    louvain_partition, H
)

# Method 2: CPM overlapping communities
print("Running CPM...", flush=True)

cpm_communities = list(
    nx.community.k_clique_communities(H, 4)
)

# Count overlapping nodes
membership = {}
for cid, comm in enumerate(cpm_communities):
    for node in comm:
        membership.setdefault(node, []).append(cid)

overlapping_nodes = [
    node for node, groups in membership.items()
    if len(groups) > 1
]

# Create a disjoint projection for CPM modularity.
# Each node is assigned to its first CPM community.
# Nodes outside CPM communities become singleton communities.
cpm_partition = {}
next_id = 0

for comm in cpm_communities:
    for node in comm:
        if node not in cpm_partition:
            cpm_partition[node] = next_id
    next_id += 1

for node in H.nodes():
    if node not in cpm_partition:
        cpm_partition[node] = next_id
        next_id += 1

if H.number_of_edges() > 0:
    cpm_modularity = nx.community.modularity(
        H,
        [
            {n for n, cid in cpm_partition.items() if cid == group}
            for group in set(cpm_partition.values())
        ]
    )
else:
    cpm_modularity = 0

# Prepare comparison
results = pd.DataFrame([
    {
        "Method": "Louvain",
        "Community_Count": len(louvain_communities),
        "Average_Community_Size": sum(
            len(c) for c in louvain_communities.values()
        ) / len(louvain_communities),
        "Largest_Community": max(
            len(c) for c in louvain_communities.values()
        ),
        "Modularity": louvain_modularity,
        "Overlapping_Nodes": 0
    },
    {
        "Method": "CPM",
        "Community_Count": len(cpm_communities),
        "Average_Community_Size": (
            sum(len(c) for c in cpm_communities) /
            len(cpm_communities) if cpm_communities else 0
        ),
        "Largest_Community": (
            max(len(c) for c in cpm_communities)
            if cpm_communities else 0
        ),
        "Modularity": cpm_modularity,
        "Overlapping_Nodes": len(overlapping_nodes)
    }
])

results.to_csv(
    TAB / "community_comparison.csv", index=False
)

print("\nCOMMUNITY COMPARISON")
print(results.to_string(index=False))

# Community count chart
plt.figure(figsize=(8, 5))
plt.bar(results["Method"], results["Community_Count"])
plt.title("Community Count Comparison")
plt.xlabel("Method")
plt.ylabel("Number of Communities")
plt.tight_layout()
plt.savefig(FIG / "community_count_comparison.png", dpi=150)
plt.close()

# Modularity chart
plt.figure(figsize=(8, 5))
plt.bar(results["Method"], results["Modularity"])
plt.title("Modularity Comparison")
plt.xlabel("Method")
plt.ylabel("Modularity")
plt.tight_layout()
plt.savefig(FIG / "modularity_comparison.png", dpi=150)
plt.close()

print("\nSUCCESS: Community evaluation completed!")
print("Comparison CSV and charts saved.")

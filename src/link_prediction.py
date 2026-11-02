
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "facebook_combined.txt"
TAB = BASE / "results" / "tables"
FIG = BASE / "results" / "figures"

TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

random.seed(42)

print("Loading Facebook network...")
G = nx.read_edgelist(DATA, nodetype=int)

# Select 500 nodes
nodes = random.sample(list(G.nodes()), min(500, len(G.nodes())))
H = G.subgraph(nodes).copy()

print("Sample nodes:", H.number_of_nodes())
print("Sample edges:", H.number_of_edges())

# Split edges into training and testing
edges = list(H.edges())
random.shuffle(edges)

test_count = max(1, int(len(edges) * 0.20))
test_edges = edges[:test_count]

train_graph = H.copy()
train_graph.remove_edges_from(test_edges)

# Generate negative examples
negative_edges = [
    (u, v) for u, v in nx.non_edges(train_graph)
    if not H.has_edge(u, v)
]

negative_edges = random.sample(
    negative_edges,
    min(len(test_edges), len(negative_edges))
)

if not test_edges or not negative_edges:
    raise ValueError("Not enough test examples.")

print("Training edges:", train_graph.number_of_edges())
print("Positive test edges:", len(test_edges))
print("Negative test edges:", len(negative_edges))

# Link prediction scoring
def get_scores(method, graph, pairs):
    if method == "Common Neighbors":
        return [
            len(list(nx.common_neighbors(graph, u, v)))
            for u, v in pairs
        ]

    if method == "Jaccard":
        return [
            score for _, _, score in
            nx.jaccard_coefficient(graph, pairs)
        ]

    if method == "Adamic-Adar":
        return [
            score for _, _, score in
            nx.adamic_adar_index(graph, pairs)
        ]

# Evaluate algorithms
results = []

all_pairs = test_edges + negative_edges
labels = [1] * len(test_edges) + [0] * len(negative_edges)

for method in ["Common Neighbors", "Jaccard", "Adamic-Adar"]:

    positive_scores = get_scores(method, train_graph, test_edges)
    negative_scores = get_scores(method, train_graph, negative_edges)

    scores = positive_scores + negative_scores
    auc = roc_auc_score(labels, scores)

    ranked = sorted(
        zip(all_pairs, scores, labels),
        key=lambda x: x[1],
        reverse=True
    )

    k = len(test_edges)
    top_k = ranked[:k]

    true_positives = sum(item[2] for item in top_k)
    precision = true_positives / k
    recall = true_positives / len(test_edges)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    results.append({
        "Algorithm": method,
        "ROC_AUC": auc,
        "Precision_at_K": precision,
        "Recall_at_K": recall,
        "F1_at_K": f1
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    TAB / "link_prediction_evaluation.csv",
    index=False
)

print("\nLINK PREDICTION RESULTS")
print(results_df.to_string(index=False))

# Predict new links using Adamic-Adar
print("\nPredicting future links...")

predictions = list(nx.adamic_adar_index(H))

predictions.sort(key=lambda x: x[2], reverse=True)

top_predictions = pd.DataFrame([
    {
        "Node_1": u,
        "Node_2": v,
        "Adamic_Adar_Score": score
    }
    for u, v, score in predictions[:20]
])

top_predictions.to_csv(
    TAB / "predicted_future_links.csv",
    index=False
)

# Comparison chart
plt.figure(figsize=(9, 5))
plt.bar(results_df["Algorithm"], results_df["ROC_AUC"])
plt.title("Link Prediction Algorithm Comparison")
plt.xlabel("Algorithm")
plt.ylabel("ROC-AUC Score")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    FIG / "link_prediction_comparison.png",
    dpi=150
)

plt.close()

print("\nSUCCESS: Link prediction completed!")
print("All results saved successfully.")

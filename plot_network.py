import matplotlib.pyplot as plt
import networkx as nx
import json

G = nx.Graph()
filename = "mexeneighbor.json"
num_edge = 0
with open(filename) as f:
    m_list = json.load(f)
print(len(m_list))
for m_dict in m_list:
        for neighbor in m_list[m_dict]:
            G.add_edge(m_dict, neighbor[0], weight = neighbor[1])
            num_edge = num_edge + 1
print(num_edge/2)
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 1]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=5)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=0.5)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=0.5, alpha=0.5, edge_color="b", style="dashed"
)

# labels
nx.draw_networkx_labels(G, pos, font_size=0.2, font_family="sans-serif")

plt.axis("off")
plt.show()
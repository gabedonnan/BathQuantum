import openqaoa
k = 3
N = 4
import networkx as nx
import matplotlib.pyplot as plt
g = nx.Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
nx.draw(g, {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 1)}, node_size=1000)
g2 = nx.Graph()
initial_edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
q = lambda u, i: u*k + i
for u in range(N):
    for i in range(k):
        for j in range(k):
            if i != j:
                g2.add_edge(q(u, i), q(u, j))

new_edges = [ (q(u, i), q(v, i)) for u, v in initial_edges for i in range(k)]
g2.add_edges_from(new_edges)
pos = {0: (0, 0), 1: (0, 2), 2: (2, 1), 3: (5, 1)}
new_pos = {q(u, i): (pos[u][0] + i*0.2**(i-1), pos[u][1]+ i*0.2) for u in range(N) for i in range(k)}
colors = ['red', 'blue', 'green', 'yellow']
nx.draw(g2, new_pos, node_color= [colors[i] for _ in range(N) for i in range(k)])
from openqaoa.problems import QUBO
import numpy as np

class GraphNode:
    def __init__(self, num):
        self.connections = {}
        self.num = num
    
    def add_connection(self, other):
        if other == None:
            return
        self.connections[other] = -1 # No colour here
        other.connections[self] = -1

    def colour(self, other, col):
        if other in self.connections:
            self.connections[other] = col
            other.connections[self] = col
        else:
            raise IndexError("connection does not exist")
    
    def get_colours(self):
        return self.connections.values()
    
    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.num == other.num
        else:
            return False
    
    def __ne__(self, other):
        if isinstance(other, GraphNode):
            return self.num != other.num
        else:
            return False
    
    def __hash__(self):
        return hash(self.num)
    
    def __repr__(self):
        return f"GN: {self.num}"
        
class Graph: # Class to make graph of size n. Used for edge colouring problem
    def __init__(self, num_nodes):
        self.nodes = [GraphNode(i) for i in range(num_nodes)]
    
    def connect_all(self):
        for node in self.nodes:
            for other in [n for n in self.nodes if n != node]:
                node.add_connection(other)
    
    def add_connections(self, edges):
        for edge in edges:
            self.nodes[edge[0]].add_connection(self.nodes[edge[1]])
        
    def colour_connection(self, num1, num2, col):
        self.nodes[num1].colour(self.nodes[num2], col)
        
    def as_qubo(self):
        A = 10
        terms = []
        final = np.zeros((len(self.nodes), len(self.nodes)))
        for i,node in enumerate(self.nodes):
            for j, link in enumerate(node.connections):
                final[i,j] += A * node.connections[link]
                final[j,i] += node.connections[link] * (A / 4)
                terms.append((node.num, link.num))
        weights = [sum(final[n % len(final)]) / (A / 4) for n in range(len(terms))]
        return QUBO(len(terms), terms, weights)
    
    
    
    
def graph_coloring_qubo(G, k, A=10) -> QUBO:
    """
    Returns the QUBO encoding of the graph coloring problem.
    
    Parameters
    ----------
    G: networkx.Graph
        The graph to be colored.
    k: int
        The number of colors to be used.
    A: float
        The weight of the terms

    Returns
    -------
    Q: dict
        The QUBO encoding of the graph coloring problem.
    """

    n_nodes = G.number_of_nodes()
    new_n_nodes = k*n_nodes

    j = lambda v,i: v*k + i # j(v,i) is the index of the new nodes corresponding to the old node v and color i

    # create matrix Q
    matrix_Q = np.zeros((new_n_nodes, new_n_nodes))
    for u, v in g.edges():
        for i in range(k):
            matrix_Q[j(u, i), j(v, i)] += A
            matrix_Q[j(v, i), j(u, i)] += A

    for u in g.nodes():
        for i, ii in nx.generators.complete_graph(n=k).edges():
            matrix_Q[j(u, i), j(u, ii)] += A
            matrix_Q[j(u, ii), j(u, i)] += A

    for u in g.nodes():
        for i in range(k):
            matrix_Q[j(u, i), j(u, i)] -= A
            
    # transform Q to J and h (bits to Ising)
    matrix_J = np.zeros((new_n_nodes, new_n_nodes))
    list_h = np.zeros(new_n_nodes)
    for i in range(new_n_nodes): 

        list_h[i] = - (matrix_Q[i, i] + sum([matrix_Q[i, j] for j in range(new_n_nodes)])) / 4

        for j in range(new_n_nodes):
            if i != j:
                matrix_J[i,j] = matrix_Q[i, j] / 4

    # get the terms and weights of the QUBO object
    terms = []
    weights = []

    for i in range(new_n_nodes):

        terms.append([i])
        weights.append(list_h[i])

        for j in range(i+1, new_n_nodes):
            if matrix_J[i,j] != 0:
                terms.append([i, j])
                weights.append(matrix_J[i,j])

    #return the QUBO object
    return QUBO(new_n_nodes, terms, weights)


x = Graph(10) #Designed to solve edge colouring problem however we did not have time to test on the actual problem, produces wrong solution for node colouring
x.connect_all()
#for n in x.nodes:
#    print(n.connections)
gc_qubo = x.as_qubo()
from openqaoa.utilities import ground_state_hamiltonian
cl_energy, cl_solutions_bitstrings = ground_state_hamiltonian(gc_qubo.hamiltonian)

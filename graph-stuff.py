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

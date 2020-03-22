from collections import deque

class DirectedAcyclicGraph:
    """
    Adjacency list representation of a DAG.
    """
    def __init__(self, vertices):
        """
        Initializes a DAG with the specified vertices and no edges.

        vertices (list or int):
            - If vertices is a list, the graph will vertices whose names are specified by the values in the list.
            - If vertices is an integer, the graph will have <vertices> nodes, numbered from 0 to vertices-1
        """
        if isinstance(vertices, int):
            vertices = range(vertices)
        
        self.vertices = {name: set() for name in vertices}  # Maps vertex name to set of outgoing edges

    def V(self):
        """
        Number of vertices in graph
        """
        return len(self.vertices)

    def E(self):
        """
        Number of directed edges in graph
        """
        return sum([len(neighbors) for neighbors in self.vertices.values()])

    def neighbors(self, u):
        """
        Returns a set of neighbors of vertex u.
        """
        return self.vertices[u]

    def add_edge(self, u, v):
        """
        Adds a directed edge from u -> v.
        """
        if u < self.V() and v < self.V():
            self.vertices[u].add(v)

    def remove_edge(self, u, v):
        """
        Removes the directed edge u -> v.
        """
        if u < self.V() and v < self.V():
            self.vertices[u].remove(v)

    def topo_sort(self):
        """
        Returns a list of vertex names for this graph in topological order
            (i.e. ancestors must come before their descendants).
        """
        in_vertices = set() # All vertices that have incoming edges
        for v in self.vertices:
            in_vertices.update(self.neighbors(v))
        start_vertices = [v for v in self.vertices if not v in in_vertices]

        visited = {}
        postorder = []
        for v in start_vertices:
            if not v in visited:
                self._dfs(v, visited, postorder)

        return postorder[::-1]

    def _dfs(self, v, visited, postorder):
        """
        (PRIVATE) Helper method to do DFS starting from node v.

        v: name of start node
        visited (dict): dictionary of already-visited nodes
        postorder (list): current postorder sequence of nodes visited; will be mutated by this method!
        """
        visited[v] = True
        for u in self.neighbors(v):
            if not u in visited:
                self._dfs(u, visited, postorder)
        postorder += [v]

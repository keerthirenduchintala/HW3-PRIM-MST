import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None
        # ensure matrix is constructed properly
        i = self.adj_mat.shape[0]
        j = self.adj_mat.shape[1]
        if i != j:
            raise ValueError 
        else:
            nodes = i
            # initialize output MST as zeros
            self.mst = np.zeros((nodes,nodes))
            # track visited nodes
            visited = set()
            #choose start node at random
            start_node = np.random.randint(0,nodes)
            # add start node to visited
            visited.add(start_node)
            # make heap que for tracking and sorting edge weight
            heap_priority_queue = []
            # initialize heap_priority_list with (weight, start, neighbor) for start_node's possible neighbors
            for neighbor in range(nodes):
                weight = self.adj_mat[start_node][neighbor]
                if weight > 0:
                    heapq.heappush(heap_priority_queue,(weight, start_node, neighbor))

            # while heap priority list queue is not empty, find the top node from heap list and add it to visited, then go to visited,
            # and add it to visited, then go to visited and add neighbors if neighbors not in visited and weight > 0 to heap que. 
            # adjust self.mst (undirected!)
            while len(heap_priority_queue) > 0:
                weight, from_node, to_node = heapq.heappop(heap_priority_queue)
                if to_node not in visited:
                    visited.add(to_node)
                    self.mst[from_node][to_node] = weight
                    self.mst[to_node][from_node] = weight
                    for neighbor in range(nodes):
                        weight = self.adj_mat[to_node][neighbor]
                        if neighbor not in visited and weight > 0:
                            heapq.heappush(heap_priority_queue,(weight,to_node, neighbor))

            




        

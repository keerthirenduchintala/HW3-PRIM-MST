import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    edge_count = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
            if mst[i,j] > 0:
                edge_count += 1
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # check that number of edges in mst is n-1
    n = mst.shape[0]
    assert edge_count == n-1, f'MST does not have {n-1} edges, has {edge_count} edges!' 

    # check connectivity by using BFS - starts at one node and visits all nodes it can reach, should match nodes in mst
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for neighbor in range(n):
            if mst[node][neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    assert len(visited) == n, 'MST is disconnected!!'

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_symmetry():
    """
    Unit test for symmetry of mst output    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    
    n = g.mst.shape[0]
    
    for i in range(n):
        for j in range(n):
            assert g.mst[i][j] == g.mst[j][i], f'MST not symmetric at [{i}][{j}]'


def test_mst_valid_edges():
    """
    Unit test that MST only contains edges from original graph
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    
    n = g.mst.shape[0]
    
    for i in range(n):
        for j in range(n):
            if g.mst[i][j] > 0:
                assert g.adj_mat[i][j] == g.mst[i][j], f'Edge [{i}][{j}] not in original graph'

import numpy as np
import math
import functools
from itertools import chain
import pickle
from node import *

nodes = {}
literals = None

ID = 1

def lookup_node(elements):
    
    global ID
    global nodes
    
    elements = tuple(elements)
    
    #looking for identical node
    idnode = nodes.get(elements)
    
    if not idnode:
        n = Node()
        n.elements = []
        for e in elements:
            p, s = e
            if p.type == DECOMPOSISTION:
                p = nodes.get(tuple(p.elements))
            else:
                p = literals.get(p.elements)
            if s.type == DECOMPOSISTION:
                s = nodes.get(tuple(s.elements))
            else:
                s = literals.get(s.elements)
            n.elements.append((p, s))
        nodes[tuple(n.elements)] = n
        
        idnode = n
    return idnode

def create_exactly_k(n,k):
    global literals
    global nodes

    # The pairwise merge below assumes a complete binary tree over variables.
    assert n >= 2 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    assert 0 < k <= n, f"need 0 < k <= n, got k={k}, n={n}"

    # Reset memoization so repeated calls (different n/k) cannot alias nodes.
    nodes = {}

    literals = dict(
        list(
            chain.from_iterable(
                ((i, Node(i, type=LITERAL)),
                (-i, Node(-i, type=LITERAL))) for i in range(1, n+1)
                )
            )
        )
    
    dp_prev = np.ndarray((n, k+1), dtype=Node)
    dp_prev.fill(None)
    
    for i in range(n):
        for j in range(2):
            dp_prev[i][j] = literals[-(i+1)] if not j else literals[i+1]
            
    for num_arr in (n//(2**i) for i in range(1, int(math.log2(n))+1)):
        dp_curr = np.ndarray((num_arr, k+1), dtype=Node)
        dp_curr.fill(None)
        
        for i in range(0, num_arr):
            for j in range(0, k+1):
                if n//num_arr < j:
                    break
                l = []
                for jj in range(j+1):
                    if (dp_prev[(i*2), jj] and dp_prev[(i*2)+1, j-jj]):
                        l.append((dp_prev[(i*2), jj], dp_prev[(i*2)+1, j-jj]))
                dp_curr[i, j] = lookup_node(l)
                    
        dp_prev = dp_curr
        
    return dp_curr

if __name__ == "__main__":
    n = 4
    k = 2
    alpha = create_exactly_k(n, k)[0][-1]
    with open(f"exactly_k_{n}_{k}.pkl", "wb") as f:
        pickle.dump(alpha, f)
        print(f"Saved exactly_k_{n}_{k}.pkl")
                    
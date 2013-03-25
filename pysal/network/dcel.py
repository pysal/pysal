"""
doubly connected edge list 

representation for network algorithms
"""


# example of edges from de berg fig 2.6

import  pysal as ps

import networkx as nx


class Vertex:
    """ """
    def __init__(self, coordinates, incident_edge):
        self.coordinates = coordinates
        incident_edge = incident_edge

class Face:
    """ """
    def __init__(self, outer_component=None, inner_component=None):

        self.outer_component = outer_component
        self.inner_component = inner_component

class Half_Edge:
    """ """
    def __init__(self, origin, twin, incident_face, Next, Prev):
        self.origin = origin
        self.twin = twin
        self.incident_face = incident_face
        self.Next = Next
        self.Prev = Prev


class DCEL:
    """Doubly connected edge list"""
    def __init__(self, graph):

        edges = {}
        vertices = {}
        faces = {}
        half_edges = {}

        cycles = nx.cycle_basis(graph)
        fi = 0
        for cycle in cycles:
            n = len(cycle)
            for i in range(n-1):
                e = (cycle[i], cycle[i+1])
                if e not in edges:
                    edges[e] = fi
                    twin_a = e[0], e[1]
                    twin_b = e[1], e[0]
                    if twin_a not in half_edges:
                        half_edges[twin_a] = fi
                    if twin_b not in half_edges:
                        half_edges[twin_b] = None
            e = cycle[n-1], cycle[0]
            if e not in edges:
                edges[e] = fi
            faces[fi] = e


            fi += 1

        self.edges = edges
        self.faces = faces
        self.half_edges = half_edges


if __name__ == '__main__':


    p1 = [
            [1,12],
            [6,12],
            [11,11],
            [14,13],
            [19,14],
            [22,9],
            [20,5],
            [16,0],
            [11,2],
            [5,1],
            [0,7],
            [2,9],
            [1,12]]

    h1 = [
            [3,7],
            [5,5],
            [8,5],
            [5,8],
            [3,7]
            ]

    h2 = [
            [4,10],
            [5,8],
            [8,5],
            [9,8],
            [4,10]
            ]

    h3 = [
            [12,6],
            [15,4],
            [18,5],
            [19,7],
            [17,9],
            [14,9],
            [12,6]
            ]
    # note that h1 union h2 forms a single hole in p1

    faces = [p1, h1, h2, h3]
    G = nx.Graph()

    for face in faces:
        n = len(face)
        for i in range(n-1):
            G.add_edge(tuple(face[i]), tuple(face[i+1]))


    cycles = nx.cycle_basis(G)
    # len of cycles is equal to the number of faces (not including external face

    dcel = DCEL(G)



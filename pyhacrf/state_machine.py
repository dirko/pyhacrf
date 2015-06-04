import numpy as np
from collections import defaultdict, deque

class StateMachine(object) : 
    base_shape = (50, 50)
        
    def __init__(self) :
        self.base_lattice = self.independent_lattice(self.base_shape)
        
        self.fully_connected = all(self.connected_edges(self.base_lattice))

    def independent_lattice(self, shape, lattice=None):
        """ Helper to construct the list of nodes and edges. """
        I, J = shape
        print(shape)

        if lattice :
            end_I = min(I, max(lattice[..., 3])) - 1
            end_J = min(J, max(lattice[..., 4])) - 1
            
            unvisited_nodes = deque([(i, j, s)
                                     for i in range(end_I)
                                     for j in range(end_J)
                                     for s in self.start_states])

            lattice = lattice.tolist()

        else :
            lattice = []
            unvisited_nodes = deque([(0, 0, s) for s in self.start_states])

        lattice += self.grow_independent_lattice((I,J), unvisited_nodes)

        lattice = np.array(sorted(lattice), dtype=int)


        return lattice


    def grow_dependent_lattice(self, x, lattice) :

        I, J, _ = x.shape
        visited_nodes = set()

        unvisited_nodes = set()
        for edge in lattice :
            unvisited_nodes.add(tuple(edge[:3]))
            unvisited_nodes.add(tuple(edge[3:6]))
        
        unvisited_nodes = deque(unvisited_nodes)
        
        lattice = []

        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(self.transitions) :
            if callable(delta) :
                transitions_d[s0].append((s1, delta, 
                                          transition_index + self.n_states))


        while unvisited_nodes:
            i, j, s0 = unvisited_nodes.popleft()
            for s1, delta, edge_parameter_index in transitions_d[s0] :
                di, dj = delta(i, j, x)
                if i + di < I and j + dj < J:
                    dest_node = (i + di, j + dj, s1)
                    edge = (i, j, s0) + dest_node + (edge_parameter_index,)
                    lattice.append(list(edge))
                    if dest_node not in visited_nodes :
                        unvisited_nodes.append(dest_node)
                        visited_nodes.add(dest_node)


        return lattice

    def grow_independent_lattice(self, shape, unvisited_nodes) :
        I, J = shape
        visited_nodes = set()
        lattice = []

        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(self.transitions) :
            if not callable(delta) :
                di, dj = delta
                transitions_d[s0].append((s1, di, dj, 
                                          transition_index + self.n_states))


        while unvisited_nodes:
            i, j, s0 = unvisited_nodes.popleft()
            for s1, di, dj, edge_parameter_index in transitions_d[s0] :
                if i + di < I and j + dj < J:
                    dest_node = (i + di, j + dj, s1)
                    edge = (i, j, s0) + dest_node + (edge_parameter_index,)
                    lattice.append(list(edge))
                    if dest_node not in visited_nodes :
                        unvisited_nodes.append(dest_node)
                        visited_nodes.add(dest_node)


        return lattice



    def connected_edges(self, lattice) :
        I = max(lattice[..., 3])
        J = max(lattice[..., 4])

        visited_nodes = {(I, J, s) for s in xrange(self.n_states)}
        connected_edges = []

        for edge in lattice[::-1] :
            source_node, dest_node = tuple(edge[0:3]), tuple(edge[3:6])
            if dest_node in visited_nodes:
                visited_nodes.add(source_node)
                connected_edges.append(True)
            else :
                connected_edges.append(False)

        return np.array(connected_edges[::-1])

    def subset_independent_lattice(self, shape) :
        I, J = shape
            
        if I < self.base_shape[0] and J < self.base_shape[1] :
            lattice = self.base_lattice[(self.base_lattice[..., 3] < I)
                                        & (self.base_lattice[..., 4] < J)]
        elif I < self.base_shape[0] :
            lattice = self.base_lattice[self.base_lattice[..., 3] < I]
            lattice = self.independent_lattice((I,J), lattice)
        elif J < self.base_shape[1] :
            lattice = self.base_lattice[self.base_lattice[..., 4] < J]
            lattice = self.independent_lattice((I,J), lattice)
        else :
            lattice = self.independent_lattice((I,J), self.base_lattice)

        return lattice


    def _build_lattice(self, x):
        """ Helper to construct the list of nodes and edges. """
        I, J, _ = x.shape

        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(self.transitions) :
            if callable(delta) :
                transitions_d[s0].append((s1, delta, 
                                          transition_index + self.n_states))

        lattice = self.subset_independent_lattice((I,J))

        if not transitions_d :
            return lattice
        else :
            lattice = lattice.tolist()

        lattice += self.grow_dependent_lattice(x, lattice)
        lattice = np.array(sorted(lattice), dtype=int)

        if not self.fully_connected :
            lattice = lattice[self.connected_edges(lattice)]

        return lattice


class DefaultStateMachine(StateMachine) :
    def __init__(self, classes) :
        n_classes = len(classes)
        deltas = ((1,1), # Match
                  (0,1), # Insertion
                  (1,0)) # Deletion

        self.start_states = [i for i in range(n_classes)]
        self.transitions = [(i, i, delta) 
                            for delta in deltas
                            for i in range(n_classes)]

        self.states_to_classes = {i : c for i, c in enumerate(classes)}

        self.n_states = len(classes)
        
        self.base_shape = (60, 60)
        
        super(DefaultStateMachine, self).__init__()

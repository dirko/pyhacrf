import numpy as np
from collections import defaultdict, deque


class GeneralStateMachine(object):

    def __init__(self, start_states, transitions, states_to_classes):
        self._start_states = start_states
        self._transitions = transitions

        max_state = max(max(s for s, _, _ in transitions), max(s for _, s, _ in transitions)) + 1
        self.n_states = max_state
        self.n_transitions = len(transitions)
        self.states_to_classes = states_to_classes

    def _build_lattice(self, x):
        """ Construct the list of nodes and edges. """
        I, J, _ = x.shape
        start_states, transitions = self._start_states, self._transitions

        lattice = []
        transitions_d = defaultdict(list)
        for transition_index, (s0, s1, delta) in enumerate(transitions) :
            transitions_d[s0].append((s1, delta, transition_index))
        # Add start states
        unvisited_nodes = deque([(0, 0, s) for s in start_states])
        visited_nodes = set()
        n_states = self.n_states

        while unvisited_nodes:
            node = unvisited_nodes.popleft()
            lattice.append(node)
            i, j, s0 = node
            for s1, delta, transition_index in transitions_d[s0] :
                try :
                    di, dj = delta
                except TypeError :
                    di, dj = delta(i, j, x)

                if i + di < I and j + dj < J:
                    edge = (i, j, s0, i + di, j + dj, s1, transition_index + n_states)
                    lattice.append(edge)
                    dest_node = (i + di, j + dj, s1)
                    if dest_node not in visited_nodes :
                        unvisited_nodes.append(dest_node)
                        visited_nodes.add(dest_node)

        lattice.sort()

        # Step backwards through lattice and add visitable nodes to the set of nodes to keep. The rest are discarded.
        final_lattice = []
        visited_nodes = set((I-1, J-1, s) for s in xrange(n_states))

        for node in lattice[::-1]:
            if node in visited_nodes:
                final_lattice.append(node)
            elif len(node) > 3:
                source_node, dest_node = node[0:3], node[3:6]
                if dest_node in visited_nodes:
                    visited_nodes.add(source_node)
                    final_lattice.append(node)

        reversed_list = list(reversed(final_lattice))

        # Squash list
        lattice = [edge for edge in reversed_list if len(edge) > 3]
        return np.array(lattice)


class DefaultStateMachine(object):
    def __init__(self, classes):
        n_classes = len(classes)
        deltas = ((1, 1),  # Match
                  (0, 1),  # Insertion
                  (1, 0))  # Deletion
        self.start_states = [i for i in range(n_classes)]
        self.transitions = [(i, i, delta)
                            for delta in deltas
                            for i in range(n_classes)]
        self.states_to_classes = {i: c for i, c in enumerate(classes)}
        self.n_states = len(classes)
        self.n_transitions = len(self.transitions)
        self.base_shape = (60, 60)
        self.base_lattice = self._independent_lattice(self.base_shape)

    def _subset_independent_lattice(self, base_lattice, base_shape, shape):
        I, J = shape

        if I < base_shape[0] and J < base_shape[1]:
            lattice = base_lattice[(base_lattice[..., 3] < I) & (base_lattice[..., 4] < J)]
        elif I < base_shape[0]:
            lattice = base_lattice[base_lattice[..., 3] < I]
            lattice = self._independent_lattice((I, J), lattice)
        elif J < base_shape[1]:
            lattice = base_lattice[base_lattice[..., 4] < J]
            lattice = self._independent_lattice((I, J), lattice)
        else:
            lattice = self._independent_lattice((I, J), base_lattice)

        return lattice

    def _independent_lattice(self, shape, lattice=None):
        """ Helper to construct the list of nodes and edges. """
        I, J = shape

        if lattice is not None:
            end_I = min(I, max(lattice[..., 3])) - 1
            end_J = min(J, max(lattice[..., 4])) - 1
            unvisited_nodes = deque([(i, j, s)
                                     for i in range(end_I)
                                     for j in range(end_J)
                                     for s in self.start_states])
            lattice = lattice.tolist()
        else:
            lattice = []
            unvisited_nodes = deque([(0, 0, s) for s in self.start_states])
        lattice += _grow_independent_lattice(self.transitions, self.n_states, (I, J), unvisited_nodes)
        lattice = np.array(sorted(lattice), dtype=int)
        return lattice

    def _build_lattice(self, x):
        """ Construct the list of nodes and edges. """
        I, J, _ = x.shape
        lattice = self._subset_independent_lattice(self.base_lattice, self.base_shape, (I, J))
        return lattice


def _grow_independent_lattice(transitions, n_states, shape, unvisited_nodes):
    I, J = shape
    visited_nodes = set()
    lattice = []

    transitions_d = defaultdict(list)
    for transition_index, (s0, s1, delta) in enumerate(transitions):
        if not callable(delta):
            di, dj = delta
            transitions_d[s0].append((s1, di, dj,
                                      transition_index + n_states))

    while unvisited_nodes:
        i, j, s0 = unvisited_nodes.popleft()
        for s1, di, dj, edge_parameter_index in transitions_d[s0]:
            if i + di < I and j + dj < J:
                dest_node = (i + di, j + dj, s1)
                edge = (i, j, s0) + dest_node + (edge_parameter_index,)
                lattice.append(list(edge))
                if dest_node not in visited_nodes:
                    unvisited_nodes.append(dest_node)
                    visited_nodes.add(dest_node)

    return lattice

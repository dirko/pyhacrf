# Some design ideas

Some thoughts to explore how to put this program together.

## General - memory vs computation
To speed training up as much as possible, I think a lattice must be constructed for each training example and then
kept in memory. During training as much time as possible must then be spent on the calculations (obviously).

Because the lattice structure only has to be unrolled once, we can incorporate slower state machines - for example
transition edges that skip to the next word.

## State machine

A possible representation for the state machine is a list of edges and the transitions they represent:
```
[('delete', 'match', (1, 1)),
 ('match', 'delete', (1, 0)),
 ('match', 'insert', (0, 1))]
```
(1, 1) means move one up and one left in the lattice. 'match' is the name of the state where you end up after doing a
'matching' transition.

To represent transitions that skip to the next word a possibility is using functions that take the current position in
the lattice and the input features and returns a transition tuple:
```
def skipi(i, j, features):
    """ Skips over rows in the lattice until the next while-space """
    for c in xrange(i, len(features[:, j])):
        if features[c] == 'white space':
            return (c, j)
```
and the corresponding transition tuple:
```
[('match', 'skip', skipi) [...]]
```

## Lattice

After inference we need access to the forward entries $alpha$.

Example alignment lattice for the two strings 'ka' and 'ch'.
```
 a  __8__
   /     \
 k5   7   9
  |  /
  4 6
  |/
 .1-2-3
  .   c   h
```

where:
1 - Start state. $alpha_{0 0 0}$.
2 - Insertion transition. $alpha_{0 1 i}$
3 - Node with all states. $alpha_{0 1 S}, S \in \{I, D, M\}$
4 - Deletion transition. $alpha_{1 0 d}$
5 - State nodes. $alpha_{1 0 s}, s\in \{I, D, M\}$
etc.
8 - Skip transition. $alpha_{1 2 s}$

In general we need to be able to address all of the edges and nodes, Because skip-edges are possible, we can't store
the $alphas$ in a table because we need access to both the origin node and the destination node.
A graph is thus necessary.

Each node can be represented as (i, j, s) - the position and state. Note that some positions can have many nodes with
different states.

Each edge is then (i0, j0, i1, j1, s0, e) - where e is the edge 'type'.

Every node and every edge is associated with a forward probability $alpha$.

## Dynamic programming/storing edge information

For the forward backward algorithm we have to be able to traverse the lattice. Maybe just having an ordering of
nodes/edges is enough.

```
a(2,0) (2,1) (2,2)

k(1,0) (1,1) (1,2)

.(0,0) (0,1) (0,2)
   .     c     h


(2 states - 'N': non-match, 'M': match)
forward = [
    (0, 0, 'M'),
    (0, 0, 'N'),
    (0, 0, 0, 1, 'N', 'N'),
    (0, 0, 0, 1, 'M', 'N'),
    (0, 0, 1, 0, 'N', 'N'),
    (0, 0, 1, 0, 'M', 'N'),
    (0, 0, 1, 1, 'M', 'M'),
    (0, 0, 1, 1, 'N', 'M'),
    (0, 1, 'M'),
    (0, 1, 'N'),
    (0, 1, 0, 2, 'N', 'N'),
    (0, 1, 0, 2, 'M', 'N'),
    (0, 1, 1, 1, 'N', 'N'),
    (0, 1, 1, 1, 'M', 'N'),
    (0, 1, 2, 1, 'N', 'N'),  # Skip
    (0, 1, 2, 1, 'M', 'N'),  # Skip

    ...[etc]...
]
```
By just iterating the list and adding the $alpha$s we fill the forward probabilities - guessing the same can be done for
the backward probabilities by iterating the list in reverse order.

## Classes

We need a lattice for each class or a way to link each node and edge with a class. Although the second option sounds
more general the start state must then belong to one class, which breaks te symmetry. Possible solution is to
allow more than one start state.

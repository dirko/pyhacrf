# Some design ideas

To speed training up as much as possible, I think a lattice must be constructed for each training example and then
kept in memory. During training as much time as possible must then be spent on the calculations (obviously).

Because the lattice structure only has to be unrolled once, we can incorporate slower state machines - for example
transition edges that skip to the next word.

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




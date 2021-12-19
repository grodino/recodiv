def normalise(d):
    """Force the sum of weights from a node d to be unitary"""

    s = sum(d.values())

    # if s == 0 it means that the node has no connection with a certain set
    if s != 0:
        for x in d.items():
            d[x[0]] = x[1] / float(s)


def normalise_by(d, s):
    for x in d.items():
        d[x[0]] = x[1] / float(s)


def add_default(d, k, n):
    if k in d:
        d[k] += n
    else:
        d[k] = n

from numpy import full, inf, zeros
from scipy.sparse.csgraph import shortest_path


def test_shortest_path_scipy_example():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
    nd = zeros((4, 4))
    nd[0, 1] = 1
    nd[0, 2] = 2
    nd[1, 3] = 1
    nd[2, 0] = 2
    nd[2, 3] = 3
    for method in ["FW", "D", "BF", "J", "auto"]:
        dist, pred = shortest_path(
            nd, method=method, directed=False, return_predecessors=True
        )
        print(dist)
        print(pred)
        assert dist[0][0] == 0
        assert dist[0][1] == 1
        assert dist[0][2] == 2
        assert dist[0][3] == 2
        assert dist[1][0] == 1
        assert dist[1][1] == 0
        assert dist[1][2] == 3
        assert dist[1][3] == 1
        assert dist[2][0] == 2
        assert dist[2][1] == 3
        assert dist[2][2] == 0
        assert dist[2][3] == 3
        assert dist[3][0] == 2
        assert dist[3][1] == 1
        assert dist[3][2] == 3
        assert dist[3][3] == 0
        assert pred[0][0] == -9999
        assert pred[0][1] == 0
        assert pred[0][2] == 0
        assert pred[0][3] == 1
        assert pred[1][0] == 1
        assert pred[1][1] == -9999
        assert pred[1][2] == 0
        assert pred[1][3] == 1
        assert pred[2][0] == 2
        assert pred[2][1] == 0
        assert pred[2][2] == -9999
        assert pred[2][3] == 2
        assert pred[3][0] == 1
        assert pred[3][1] == 3
        assert pred[3][2] == 3
        assert pred[3][3] == -9999


def test_shortest_path_wikipedia_example():
    # https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    nd = full((4, 4), inf)
    nd[0, 2] = -2
    nd[1, 0] = 4
    nd[1, 2] = 3
    nd[2, 3] = 2
    nd[3, 1] = -1
    nd[0, 0] = 0
    nd[1, 1] = 0
    nd[2, 2] = 0
    nd[3, 3] = 0
    # Skip "D" on this one because of negative weights
    for method in ["FW", "BF", "J", "auto"]:
        print(method)
        print("before")
        print(nd)
        dist, pred = shortest_path(
            nd, method=method, directed=True, return_predecessors=True
        )
        print("after")
        print(dist)
        print(pred)
        assert dist[0][0] == 0
        assert dist[0][1] == -1
        assert dist[0][2] == -2
        assert dist[0][3] == 0
        assert dist[1][0] == 4
        assert dist[1][1] == 0
        assert dist[1][2] == 2
        assert dist[1][3] == 4
        assert dist[2][0] == 5
        assert dist[2][1] == 1
        assert dist[2][2] == 0
        assert dist[2][3] == 2
        assert dist[3][0] == 3
        assert dist[3][1] == -1
        assert dist[3][2] == 1
        assert dist[3][3] == 0
        assert pred[0][0] == -9999
        assert pred[0][1] == 3
        assert pred[0][2] == 0
        assert pred[0][3] == 2
        assert pred[1][0] == 1
        assert pred[1][1] == -9999
        assert pred[1][2] == 0
        assert pred[1][3] == 2
        assert pred[2][0] == 1
        assert pred[2][1] == 3
        assert pred[2][2] == -9999
        assert pred[2][3] == 2
        assert pred[3][0] == 1
        assert pred[3][1] == 3
        assert pred[3][2] == 0
        assert pred[3][3] == -9999

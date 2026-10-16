import numpy as np
from gp.tree import Individual, Node, get_all_nodes


def test_tree_depth_and_size():
    # Tree: (x0 + (x1 * 3.0))
    leaf1 = Node("x0")
    leaf2 = Node("x1")
    leaf3 = Node(3.0)
    mul_node = Node("mul", [leaf2, leaf3])
    root = Node("add", [leaf1, mul_node])

    assert leaf1.depth() == 1
    assert leaf1.size() == 1
    assert mul_node.depth() == 2
    assert mul_node.size() == 3
    assert root.depth() == 3
    assert root.size() == 5


def test_tree_evaluation():
    # Tree: ((x0 + x1) / 2.0)
    leaf0 = Node("x0")
    leaf1 = Node("x1")
    add_node = Node("add", [leaf0, leaf1])
    two_node = Node(2.0)
    root = Node("div", [add_node, two_node])

    X = np.array([
        [1.0, 3.0],
        [4.0, 6.0],
        [-2.0, 2.0],
    ])
    y_pred = root.evaluate(X)
    assert np.allclose(y_pred, [2.0, 5.0, 0.0])


def test_deep_copy():
    leaf = Node("x0")
    root = Node("sin", [leaf])
    copied = root.copy()

    assert copied.value == root.value
    assert copied is not root
    assert copied.children[0] is not root.children[0]

    # Modifying copy must not affect original
    copied.value = "cos"
    assert root.value == "sin"


def test_get_all_nodes_and_canonical_repr():
    leaf1 = Node("x0")
    leaf2 = Node(1.5)
    root = Node("add", [leaf1, leaf2])

    all_nodes = get_all_nodes(root)
    assert len(all_nodes) == 3
    assert root.canonical_repr() == "add(x0,C(1.5))"

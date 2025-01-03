from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Set
from graphviz import Digraph


def main():
    print(f"Hello there {__name__}")


@dataclass(frozen=True)
class Value:
    data: float
    _children: Tuple["Value", ...] = field(default_factory=tuple)
    _op: str = ""
    label: str = ""
    grad: float = 0.0

    @property
    def _prev(self) -> Set["Value"]:
        return set(self._children)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other: "Value") -> "Value":
        if not isinstance(other, Value):
            raise TypeError("Operands must be instances of Value")
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other: "Value") -> "Value":
        if not isinstance(other, Value):
            raise TypeError("Operands must be instances of Value")
        return Value(self.data * other.data, (self, other), "*")

    def with_label(self, label: str) -> "Value":
        # Create a new Value object with the same properties, but with a new label
        return Value(self.data, self._children, self._op, label, self.grad)


def trace(root: Value):
    # Builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}",
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


if __name__ == "__main__":
    print(Value(2.0))
    main()

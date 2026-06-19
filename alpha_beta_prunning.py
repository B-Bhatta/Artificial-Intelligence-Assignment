r"""
Alpha-Beta Pruning Demo

Run:
    python alpha_beta_prunning.py

This program:
1. Builds a small game tree.
2. Applies Alpha-Beta Pruning.
3. Shows the optimal value.
4. Displays the pruned structural connections.
5. Generates an SVG visualization file.
"""

from dataclasses import dataclass, field
from math import inf
from typing import Dict, List, Optional, Tuple


@dataclass
class Node:
    name: str
    node_type: str  # "MAX", "MIN", or "LEAF"
    value: Optional[int] = None
    children: List[str] = field(default_factory=list)
    calculated_value: Optional[int] = None
    visited: bool = False


Tree = Dict[str, Node]
PrunedEdge = Tuple[str, str]


def build_game_tree() -> Tree:
    r"""
    Tree structure:

                    A(MAX)
                  /        \
              B1(MIN)     B2(MIN)
             /     \       /     \
          C1(MAX) C2(MAX) C3(MAX) C4(MAX)
          /  \     /  \    /  \    /  \
         3    5   4    9  5    4  100 200

    Alpha-beta pruning will prune the structural connection: B2 -> C4
    """
    return {
        "A": Node("A", "MAX", children=["B1", "B2"]),
        "B1": Node("B1", "MIN", children=["C1", "C2"]),
        "B2": Node("B2", "MIN", children=["C3", "C4"]),
        "C1": Node("C1", "MAX", children=["D1", "D2"]),
        "C2": Node("C2", "MAX", children=["D3", "D4"]),
        "C3": Node("C3", "MAX", children=["D5", "D6"]),
        "C4": Node("C4", "MAX", children=["D7", "D8"]),
        "D1": Node("D1", "LEAF", value=3),
        "D2": Node("D2", "LEAF", value=5),
        "D3": Node("D3", "LEAF", value=4),
        "D4": Node("D4", "LEAF", value=9),
        "D5": Node("D5", "LEAF", value=5),
        "D6": Node("D6", "LEAF", value=4),
        "D7": Node("D7", "LEAF", value=100),
        "D8": Node("D8", "LEAF", value=200),
    }


def alpha_beta(
    tree: Tree,
    node_name: str,
    alpha: float,
    beta: float,
    pruned_edges: List[PrunedEdge],
) -> int:
    """Apply Alpha-Beta Pruning and return the best minimax value."""
    node = tree[node_name]
    node.visited = True

    if node.node_type == "LEAF":
        node.calculated_value = node.value
        return node.value  # type: ignore[return-value]

    if node.node_type == "MAX":
        best_value = -inf

        for index, child_name in enumerate(node.children):
            value = alpha_beta(tree, child_name, alpha, beta, pruned_edges)
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)

            # Cut-off condition for MAX node
            if beta <= alpha:
                for remaining_child in node.children[index + 1 :]:
                    pruned_edges.append((node.name, remaining_child))
                break

        node.calculated_value = int(best_value)
        return int(best_value)

    if node.node_type == "MIN":
        best_value = inf

        for index, child_name in enumerate(node.children):
            value = alpha_beta(tree, child_name, alpha, beta, pruned_edges)
            best_value = min(best_value, value)
            beta = min(beta, best_value)

            # Cut-off condition for MIN node
            if beta <= alpha:
                for remaining_child in node.children[index + 1 :]:
                    pruned_edges.append((node.name, remaining_child))
                break

        node.calculated_value = int(best_value)
        return int(best_value)

    raise ValueError(f"Unknown node type: {node.node_type}")


def generate_svg(tree: Tree, pruned_edges: List[PrunedEdge], filename: str = "alpha_beta_tree.svg") -> None:
    """Generate a simple SVG visualization without using external libraries."""

    positions = {
        "A": (500, 50),
        "B1": (250, 160),
        "B2": (750, 160),
        "C1": (125, 280),
        "C2": (375, 280),
        "C3": (625, 280),
        "C4": (875, 280),
        "D1": (70, 410),
        "D2": (180, 410),
        "D3": (320, 410),
        "D4": (430, 410),
        "D5": (570, 410),
        "D6": (680, 410),
        "D7": (820, 410),
        "D8": (930, 410),
    }

    pruned_set = set(pruned_edges)

    def node_label(node: Node) -> str:
        if node.node_type == "LEAF":
            return f"{node.name}\n{node.value}"
        if node.calculated_value is not None:
            return f"{node.name}\n{node.node_type}={node.calculated_value}"
        return f"{node.name}\n{node.node_type}"

    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520" viewBox="0 0 1000 520">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="25" text-anchor="middle" font-size="20" font-family="Arial" font-weight="bold">Alpha-Beta Pruning Tree</text>',
        '<text x="500" y="500" text-anchor="middle" font-size="14" font-family="Arial">Dashed red connection means pruned branch</text>',
    ]

    # Draw edges first
    for parent_name, parent in tree.items():
        if not parent.children:
            continue
        x1, y1 = positions[parent_name]
        for child_name in parent.children:
            x2, y2 = positions[child_name]
            if (parent_name, child_name) in pruned_set:
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1 + 25}" x2="{x2}" y2="{y2 - 25}" '
                    f'stroke="red" stroke-width="3" stroke-dasharray="8,6"/>'
                )
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                svg_parts.append(
                    f'<text x="{mid_x}" y="{mid_y}" text-anchor="middle" font-size="12" '
                    f'font-family="Arial" fill="red">PRUNED</text>'
                )
            else:
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1 + 25}" x2="{x2}" y2="{y2 - 25}" '
                    f'stroke="black" stroke-width="2"/>'
                )

    # Draw nodes
    for node_name, node in tree.items():
        x, y = positions[node_name]
        is_pruned_node = any(child == node_name for _, child in pruned_edges) or (
            node_name in ["D7", "D8"]
        )
        fill_color = "#ffecec" if is_pruned_node else "#eef5ff"
        stroke_color = "red" if is_pruned_node else "black"
        opacity = "0.55" if is_pruned_node else "1.0"

        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="34" fill="{fill_color}" stroke="{stroke_color}" '
            f'stroke-width="2" opacity="{opacity}"/>'
        )

        lines = node_label(node).split("\n")
        text_y = y - 5 if len(lines) == 2 else y
        for i, line in enumerate(lines):
            svg_parts.append(
                f'<text x="{x}" y="{text_y + i * 16}" text-anchor="middle" '
                f'font-size="13" font-family="Arial" font-weight="bold">{line}</text>'
            )

    svg_parts.append("</svg>")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("\n".join(svg_parts))


def main() -> None:
    tree = build_game_tree()
    pruned_edges: List[PrunedEdge] = []

    print("Executing Alpha-Beta Search...")
    optimal_value = alpha_beta(tree, "A", -inf, inf, pruned_edges)
    print(f"Alpha-Beta execution completed. Optimal value: {optimal_value}")
    print(f"Pruned structural connections detected: {pruned_edges}")

    output_svg = "alpha_beta_tree.svg"
    generate_svg(tree, pruned_edges, output_svg)
    print(f"Tree visual representation generated successfully as: {output_svg}")


if __name__ == "__main__":
    main()

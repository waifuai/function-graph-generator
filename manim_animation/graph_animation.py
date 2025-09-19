from manim import *

class FunctionDependencyGraph(Scene):
    def construct(self):
        # Title
        title = Text("Function Dependency Graph Generation", font_size=24)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # Section 1: Graph Theory Preliminaries
        section1 = Text("1. Graph Theory Preliminaries", font_size=20, color=BLUE)
        self.play(Write(section1))
        self.wait(1)
        self.play(section1.animate.to_edge(LEFT).shift(UP * 0.5))

        # Explanation of vertices and edges
        vertices_text = Text("Vertices (V): Represent functions", font_size=16)
        edges_text = Text("Edges (E): Represent function calls", font_size=16)

        vertices_text.next_to(section1, DOWN, buff=0.5)
        edges_text.next_to(vertices_text, DOWN)

        self.play(Write(vertices_text))
        self.play(Write(edges_text))
        self.wait(2)

        # Clear for next section
        self.play(FadeOut(section1), FadeOut(vertices_text), FadeOut(edges_text))

        # Section 2: Graph Construction
        section2 = Text("2. Graph Construction Algorithm", font_size=20, color=BLUE)
        self.play(Write(section2))
        self.wait(1)
        self.play(section2.animate.to_edge(LEFT).shift(UP * 0.5))

        # Show function identification
        func_id_text = Text("Step 1: Function Identification", font_size=16)
        func_id_text.next_to(section2, DOWN, buff=0.5)
        self.play(Write(func_id_text))

        # Create sample functions as circles
        functions = ["foo", "bar", "baz"]
        circles = VGroup(*[Circle(radius=0.5, color=WHITE) for _ in functions])
        labels = VGroup(*[Text(func, font_size=16) for func in functions])

        for i, (circle, label) in enumerate(zip(circles, labels)):
            circle.move_to(RIGHT * (i - 1) * 2)
            label.move_to(circle.get_center())

        self.play(Create(circles), Write(labels))
        self.wait(2)

        # Step 2: Call Identification
        call_id_text = Text("Step 2: Call Identification", font_size=16)
        call_id_text.next_to(func_id_text, DOWN, buff=0.5)
        self.play(Write(call_id_text))

        # Add edges for function calls
        edges = [
            Arrow(circles[0].get_right(), circles[1].get_left(), color=YELLOW),
            Arrow(circles[0].get_bottom(), circles[2].get_top(), color=YELLOW),
            Arrow(circles[1].get_bottom(), circles[2].get_top(), color=YELLOW)
        ]

        for edge in edges:
            self.play(Create(edge))
            self.wait(0.5)

        self.wait(2)

        # Clear for next section
        self.play(FadeOut(section2), FadeOut(func_id_text), FadeOut(call_id_text),
                  FadeOut(circles), FadeOut(labels), *[FadeOut(edge) for edge in edges])

        # Section 3: Graph Properties
        section3 = Text("3. Graph Properties", font_size=20, color=BLUE)
        self.play(Write(section3))
        self.wait(1)
        self.play(section3.animate.to_edge(LEFT).shift(UP * 0.5))

        # Create a more complex graph for properties
        complex_graph = Graph(
            vertices=[1, 2, 3, 4, 5],
            edges=[(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (5, 1)],  # Creating a cycle
            layout="circular",
            vertex_config={"fill_color": WHITE, "radius": 0.3},
            edge_config={"color": WHITE}
        )

        self.play(Create(complex_graph))
        self.wait(1)

        # Highlight cycle
        cycle_text = Text("Cycle: 1 → 2 → 4 → 5 → 1", font_size=16, color=RED)
        cycle_text.next_to(section3, DOWN, buff=0.5)
        self.play(Write(cycle_text))

        # Highlight the cycle edges
        cycle_edges = [(1, 2), (2, 4), (4, 5), (5, 1)]
        for edge in cycle_edges:
            self.play(complex_graph.edges[edge].animate.set_color(RED))
            self.wait(0.5)

        self.wait(2)

        # Show strongly connected component
        scc_text = Text("Strongly Connected Component", font_size=16, color=GREEN)
        scc_text.next_to(cycle_text, DOWN)
        self.play(Write(scc_text))

        # Highlight SCC (nodes 1, 2, 4, 5)
        scc_nodes = [1, 2, 4, 5]
        for node in scc_nodes:
            self.play(complex_graph.vertices[node].animate.set_fill(GREEN))
            self.wait(0.5)

        self.wait(2)

        # Show degrees
        degree_text = Text("Node Degrees", font_size=16)
        degree_text.next_to(scc_text, DOWN)
        self.play(Write(degree_text))

        # Annotate degrees for a few nodes
        node1_degree = Text("Node 1: In=1, Out=2", font_size=12)
        node1_degree.next_to(complex_graph.vertices[1], UP)
        self.play(Write(node1_degree))

        node5_degree = Text("Node 5: In=2, Out=1", font_size=12)
        node5_degree.next_to(complex_graph.vertices[5], DOWN)
        self.play(Write(node5_degree))

        self.wait(3)

        # Conclusion
        conclusion = Text("Function dependency graphs help analyze code structure,\nidentify cycles, and understand connectivity.", font_size=16)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))

        self.wait(3)

        # Fade out everything
        self.play(FadeOut(section3), FadeOut(complex_graph), FadeOut(cycle_text),
                  FadeOut(scc_text), FadeOut(degree_text), FadeOut(node1_degree),
                  FadeOut(node5_degree), FadeOut(conclusion), FadeOut(title))

if __name__ == "__main__":
    pass
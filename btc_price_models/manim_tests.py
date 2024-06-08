

# %%

from manim import *

class ExponentialFunction(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
        )

        labels = axes.get_axis_labels(x_label="t", y_label="y(t)")

        # Function definition for different r values
        r_values = [0.5, 1, 1.5]
        colors = [YELLOW, GREEN, RED]
        graphs = [
            axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]
        graph_labels = [
            axes.get_graph_label(graph, label=f"r={r}", x_val=2, direction=UP + LEFT)
            for graph, r in zip(graphs, r_values)
        ]

        # Title and explanation text
        title = Title("Exponential Function: \( y(t) = e^{rt} \)", include_underline=False)
        explanation_text = Text(
            "As r increases, the exponential growth rate increases",
            font_size=24,
        ).to_edge(DOWN)

        # Animation
        self.add(title)
        self.play(Create(axes), Write(labels))
        self.wait(1)
        for graph, graph_label in zip(graphs, graph_labels):
            self.play(Create(graph), FadeIn(graph_label, shift=UP))
            self.wait(1)
        self.play(Write(explanation_text))
        self.wait(2)

# To run this, use:
# manim -pql script_name.py ExponentialFunction
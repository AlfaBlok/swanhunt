from manim import *

class DefaultTemplate(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation

class ExponentialFunction(Scene):
    def construct(self):
        # Adjust the overall layout
        title = Title("Función Exponencial: \( y(t) = e^{rt} \)", include_underline=False).to_edge(UP)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=10, 
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(x_label="t", y_label="y(t)")

        # Define the exponential functions with different growth rates
        r_values = [0.5, 1, 1.5]
        colors = [YELLOW, GREEN, RED]
        graphs = [
            axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]

        # Position labels inside the chart area
        label_positions = [(3.5, 10), (4.1, 35), (1.8, 40)]
        graph_labels = [
            axes.get_graph_label(graph, label=f"r={r}", x_val=2)
            .move_to(axes.c2p(*pos))
            for graph, r, pos in zip(graphs, r_values, label_positions)
        ]

        # Explanation text
        explanation_text = Text(
            "Según r crece el crecimiento exponencial  acelera",
            font_size=24,
        ).to_edge(DOWN)

        # Animate
        self.add(title)
        self.play(Create(axes), Write(labels))
        self.wait(1)
        for graph, graph_label in zip(graphs, graph_labels):
            self.play(Create(graph), FadeIn(graph_label, shift=UP))
            self.wait(1)
        self.play(Write(explanation_text))
        self.wait(2)

from manim import *

class ExponentialFunction2(Scene):
    def construct(self):
        # Adjust the overall layout
        title = Title("Función Exponencial: \( y(t) = e^{rt} \)", include_underline=False).to_edge(UP)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=10, 
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(x_label="t", y_label="rt")

        # Define the exponential functions with different growth rates
        r_values = [0.5, 1, 1.5]
        colors = [YELLOW, GREEN, RED]
        exponential_graphs = [
            axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]

        # Position labels inside the chart area
        label_positions = [(3.5, 10), (4.1, 35), (1.8, 40)]
        label_positions2 = [(3.5, 5), (4.1, 17), (1.8, 20)]
        
        graph_labels = [
            axes.get_graph_label(exponential_graphs[i], label=f"r={r}", x_val=2)
            .move_to(axes.c2p(*pos))
            for i, (r, pos) in enumerate(zip(r_values, label_positions))
        ]

        graph_labels2 = [
            axes.get_graph_label(exponential_graphs[i], label=f"r={r}", x_val=2)
            .move_to(axes.c2p(*pos))
            for i, (r, pos) in enumerate(zip(r_values, label_positions2))
        ]

        # Define the logarithmic graphs
        log_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=10,
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)
        logarithmic_graphs = [
            log_axes.plot(lambda t: r * t, color=color)
            for r, color in zip(r_values, colors)
        ]

        # Explanation text
        explanation_text = Text(
            "Según r crece el crecimiento exponencial acelera",
            font_size=24,
        ).to_edge(DOWN)

        # Explanation text 2 for logarithmic phase
        explanation_text2 = Text(
            "El ritmo de exponenciación es constante",
            font_size=24,
        ).to_edge(DOWN)

        # Animate
        self.add(title)
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # Group each graph with its label and animate creation together
        for graph, graph_label in zip(exponential_graphs, graph_labels):
            graph_group = VGroup(graph, graph_label)
            self.play(Create(graph_group), run_time=1)
            self.wait(0.1)

        self.play(Write(explanation_text))
        self.wait(1)

        # Change the title to indicate logarithmic transformation
        new_title = Title("Función Log(Exponencial): \( \log(y(t)) = log(e^{rt}) \)", include_underline=False).to_edge(UP)
        self.play(Transform(title, new_title))

        # # Transform each graph and its label to logarithmic versions together
        # for exp_graph, log_graph, graph_label, graph_label2 in zip(exponential_graphs, logarithmic_graphs, graph_labels, graph_labels2):
        #     self.play(Transform(exp_graph, log_graph), Transform(graph_label, graph_label2), run_time=1)
        # self.wait(1)

        original_groups = [
            VGroup(exp_graph, graph_label)
            for exp_graph, graph_label in zip(exponential_graphs, graph_labels)
        ]

        target_groups = [
            VGroup(log_graph, graph_label2)
            for log_graph, graph_label2 in zip(logarithmic_graphs, graph_labels2)
        ]

        # Create a single animation where all transformations are done simultaneously
        transformations = [
            Transform(orig_group, target_group)
            for orig_group, target_group in zip(original_groups, target_groups)
        ]

        self.play(*transformations, run_time=2)

        # Transform the explanation text
        self.play(Transform(explanation_text, explanation_text2))
        self.wait(3)




class ExponentialFunction3(Scene):
    def construct(self):
        title = Title("Función Exponencial: \( y(t) = e^{rt} \)", include_underline=False).to_edge(UP)
        self.add(title)

        # Original Axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=10, 
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)
        labels = axes.get_axis_labels(x_label="t", y_label="rt")
        self.play(Create(axes), Write(labels))

        # Original Graphs
        r_values = [0.5, 1, 1.5]
        colors = [YELLOW, GREEN, RED]
        exponential_graphs = [
            axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]
        for graph in exponential_graphs:
            self.play(Create(graph), run_time=1)

        # New Axes
        new_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],  # New y-range
            x_length=10,
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)

        # New Graphs
        new_graphs = [
            new_axes.plot(lambda t: np.exp(r * t) / 10, color=color)  # Adjust function to fit new y-range
            for r, color in zip(r_values, colors)
        ]

        # Group old axes and graphs
        old_group = VGroup(axes, *exponential_graphs)
        new_group = VGroup(new_axes, *new_graphs)

        # Simultaneous Transformation
        self.play(Transform(old_group, new_group), run_time=2)
        self.wait(2)



class ExponentialFunction4(Scene):
    def construct(self):
        # Adjust the overall layout
        title = Title("Función Exponencial: \( y(t) = e^{rt} \)", include_underline=False).to_edge(UP)
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=10, 
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(x_label="t", y_label="rt")

        # Define the exponential functions with different growth rates
        r_values = [0.5, 1, 1.5]
        colors = [YELLOW, GREEN, RED]
        exponential_graphs = [
            axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]

        # Position labels inside the chart area
        label_positions = [(3.5, 10), (4.1, 35), (1.8, 40)]
        label_positions2 = [(3.5, 5), (4.1, 17), (1.8, 20)]
        
        graph_labels = [
            axes.get_graph_label(exponential_graphs[i], label=f"r={r}", x_val=2)
            .move_to(axes.c2p(*pos))
            for i, (r, pos) in enumerate(zip(r_values, label_positions))
        ]

        graph_labels2 = [
            axes.get_graph_label(exponential_graphs[i], label=f"r={r}", x_val=2)
            .move_to(axes.c2p(*pos))
            for i, (r, pos) in enumerate(zip(r_values, label_positions2))
        ]

        # Define the logarithmic graphs
        log_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=10,
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)
        logarithmic_graphs = [
            log_axes.plot(lambda t: r * t, color=color)
            for r, color in zip(r_values, colors)
        ]

        # Explanation text
        explanation_text = Text(
            "Según r crece el crecimiento exponencial acelera",
            font_size=24,
        ).to_edge(DOWN)

        # Explanation text 2 for logarithmic phase
        explanation_text2 = Text(
            "El ritmo de exponenciación es constante",
            font_size=24,
        ).to_edge(DOWN)


        self.add(title)
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # Group each graph with its label and animate creation together
        for graph, graph_label in zip(exponential_graphs, graph_labels):
            graph_group = VGroup(graph, graph_label)
            self.play(Create(graph_group), run_time=1)
            self.wait(0.1)

        self.play(Write(explanation_text))
        self.wait(1)

        # Change the title to indicate logarithmic transformation
        new_title = Title("Función Log(Exponencial): \( \log(y(t)) = log(e^{rt}) \)", include_underline=False).to_edge(UP)
        self.play(Transform(title, new_title))

        # # Transform each graph and its label to logarithmic versions together
        # for exp_graph, log_graph, graph_label, graph_label2 in zip(exponential_graphs, logarithmic_graphs, graph_labels, graph_labels2):
        #     self.play(Transform(exp_graph, log_graph), Transform(graph_label, graph_label2), run_time=1)
        # self.wait(1)

        original_groups = [
            VGroup(exp_graph, graph_label)
            for exp_graph, graph_label in zip(exponential_graphs, graph_labels)
        ]

        target_groups = [
            VGroup(log_graph, graph_label2)
            for log_graph, graph_label2 in zip(logarithmic_graphs, graph_labels2)
        ]

        # Create a single animation where all transformations are done simultaneously
        transformations = [
            Transform(orig_group, target_group)
            for orig_group, target_group in zip(original_groups, target_groups)
        ]

        self.play(*transformations, run_time=2)

        # Transform the explanation text
        self.play(Transform(explanation_text, explanation_text2))
        self.wait(0.1)

# Clear the screen of original groups or make sure they are transformed completely
        self.remove(*[group for group, _ in zip(original_groups, target_groups)],axes, labels, explanation_text)

        # Move and scale the logarithmic graphs to the right
        log_axes_group = VGroup(log_axes, *logarithmic_graphs, *graph_labels2)
        self.play(log_axes_group.animate.scale(0.5).to_edge(RIGHT), run_time=2)
        # self.wait(1)

        # Create and display the original chart on the left, scaled down

        # Create and display the original chart on the left, scaled down
        original_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=10, 
            y_length=4,
            axis_config={"color": BLUE},
        ).shift(DOWN * 0.5)

        original_graphs = [
            original_axes.plot(lambda t: np.exp(r * t), color=color)
            for r, color in zip(r_values, colors)
        ]
        original_chart_group = VGroup(original_axes, *original_graphs, *graph_labels)

        # Position and scale the original chart group directly without additional animation
        # original_chart_group.scale(0.5)  # Scale first
        # original_chart_group.to_edge(LEFT)  # Then position to the left

        # Fade in the original chart group in its final position
        self.play(FadeIn(original_chart_group), run_time=0.5)
        self.play(original_chart_group.animate.scale(0.5).to_edge(LEFT), run_time=2)
        self.wait(2)  # Final pause to show both charts side by side



class ExponentialFunction5(Scene):
    def construct(self):
        # Titles and axes setup
        title = Title("Comparación de Funciones: Lineal y Logarítmica", include_underline=False).to_edge(UP)

        # Linear scale axes with labels
        linear_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 50, 10],
            x_length=5,
            y_length=4,
            axis_config={"color": BLUE},
            tips=True,
        ).to_edge(LEFT).shift(RIGHT / 2)
        linear_labels = linear_axes.get_axis_labels(x_label="t", y_label="y")

        # Logarithmic scale axes with labels
        log_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=4,
            axis_config={"color": BLUE},
            tips=True,
        ).to_edge(RIGHT).shift(LEFT / 2)
        log_labels = log_axes.get_axis_labels(x_label="t", y_label="log(y)")

        # Formulas under each chart
        linear_formula = MathTex("y(t) = e^{rt}", font_size=48).next_to(linear_axes, DOWN)
        logarithmic_formula = MathTex("\log(y(t)) = \log(e^{rt}) = rt", font_size=48).next_to(log_axes, DOWN)

        # r value tracker and label
        r_tracker = ValueTracker(1)
        r_label = MathTex("r = ", font_size=36)
        r_value = DecimalNumber(r_tracker.get_value(), num_decimal_places=2, font_size=36)
        r_value.next_to(r_label, RIGHT)
        r_value.add_updater(lambda d: d.set_value(r_tracker.get_value()))

        # Position r label and value in the center bottom of the screen
        r_display_group = VGroup(r_label, r_value)
        r_display_group.to_edge(DOWN/2)

        # Linear graph
        linear_graph = always_redraw(lambda: linear_axes.plot(
            lambda t: np.exp(r_tracker.get_value() * t), color=YELLOW
        ))

        # Logarithmic graph
        logarithmic_graph = always_redraw(lambda: log_axes.plot(
            lambda t: r_tracker.get_value() * t, color=GREEN
        ))

        # Display everything including the axis labels and r label
        self.add(title, linear_axes, log_axes, linear_graph, logarithmic_graph, linear_labels, log_labels, linear_formula, logarithmic_formula, r_display_group)
        
        # Example of changing r dynamically
        self.play(r_tracker.animate.set_value(1.5), run_time=2)
        self.wait(1)
        self.play(r_tracker.animate.set_value(0.5), run_time=2)
        self.wait(1)
        
        # Keeping both graphs updated with changes in r
        self.play(r_tracker.animate.set_value(2), run_time=2)
        self.wait(2)



class CompoundInterest(Scene):
    def construct(self):
        # Set up the title and explanation
        title = Title("Continuous Compounding and its Limit to \(e^r\)", font_size=24)
        self.add(title)

        # Create dynamic parts of the formula
        r = 1  # This is the interest rate used in the example
        n_tracker = ValueTracker(2)  # Starts at dividing by 2

        # Dynamic compound interest formula part
        formula_part = always_redraw(lambda: MathTex(
            "P_1", "=", "P_0", "\\left(1 + \\frac{r}{", f"{int(n_tracker.get_value())}", "}\\right)^{", f"{int(n_tracker.get_value())}", "}",
            font_size=48
        ).next_to(title, DOWN))

        # Display initial formula
        self.play(Write(formula_part))
        self.wait(1)

        # Animate n from 2 to 3 to n
        self.play(n_tracker.animate.set_value(3), run_time=2)
        self.wait(1)
        self.play(n_tracker.animate.set_value(10), run_time=2)  # Show several steps quickly
        self.wait(1)

        # Display the definition of e
        e_definition = MathTex(
            "\\lim_{n \\to \\infty} \\left(1 + \\frac{r}{n}\\right)^n", "=", "e^r",
            font_size=48
        ).next_to(formula_part, DOWN)

        self.play(Write(e_definition))
        self.wait(2)

        # Highlight that both the part of the formula and the definition of e are equivalent
        self.play(
            formula_part[-3:].animate.set_color(YELLOW),
            e_definition[0].animate.set_color(YELLOW)
        )
        self.wait(2)

        # Final hold to show the connection
        self.play(FadeOut(VGroup(title, formula_part, e_definition)))
        self.wait(1)
import numpy as np
from manim import *
import random
import colorsys
from manim import rgb_to_color
from manim.utils.space_ops import rotate_vector
import math


def evenly_spaced_colors(n):
    """
    Returns a list of `n` Color objects evenly spaced in HSV hue.
    Each color is highly saturated (S=1) and bright (V=1).
    """
    return [
        rgb_to_color(colorsys.hsv_to_rgb(i / n, 1.0, 1.0))
        for i in range(n)
    ]


class Intro(Scene):
    def construct(self):
        # Title
        title = Paragraph(
            "Základ kvantového algoritmu",
            "shlukování metodou nejbližších středů",
            font_size=42,
            line_spacing=1.5,
            alignment="center"  # zarovná každý řádek na střed
        ).center()
        # Subtitle
        subtitle = Text("Ing. Vít Nováček PhD, Michal Forgó", font_size=20).next_to(title, DOWN, buff=0.4)
        # Date
        lecture_date = Text("14.5.2025", font_size=20).to_edge(UR, buff=0.4)

        # Two images
        interreg_logo = ImageMobject("./assets/interreg.png").scale(0.5).to_edge(DL)
        ntc_logo = SVGMobject("./assets/ntc_logo.svg").scale(0.5).to_edge(DR)

        # Add everything to the scene
        self.add(title, subtitle, lecture_date, interreg_logo, ntc_logo)

        # Wait 5 seconds so the viewer can see the result
        self.wait(5)

        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(lecture_date),
            FadeOut(interreg_logo),
            FadeOut(ntc_logo)
        )
        self.wait()


class Kmeans(Scene):
    def construct(self):
        random.seed(11)
        K = random.randint(2, 4)

        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)

        self.play(Create(axes))

        NUM_POINTS = 21
        data_coords = []
        data_dots = VGroup()

        for _ in range(NUM_POINTS):
            x = random.uniform(0, 9)
            y = random.uniform(0, 9)
            dot = Dot(axes.coords_to_point(x, y), color=GREY)
            data_coords.append((x, y))
            data_dots.add(dot)

        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.5) for dot in data_dots], lag_ratio=0.1)
        )
        self.wait(1)

        cluster_coords = []
        cluster_dots = VGroup()
        cluster_colors = evenly_spaced_colors(K)

        for i in range(K):
            cx = random.uniform(0, 9)
            cy = random.uniform(0, 9)
            c_dot = Dot(
                axes.coords_to_point(cx, cy),
                color=cluster_colors[i],
                radius=0.14
            )
            cluster_coords.append((cx, cy))
            cluster_dots.add(c_dot)

        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.8) for dot in cluster_dots], lag_ratio=0.1)
        )
        self.wait(1)

        MAX_ITER = 10
        THRESHOLD = 0.02
        special_animation_done = False
        special_cluster_idx = random.randint(0, K-1)

        for iteration in range(MAX_ITER):
            if iteration == 0:
                first_index = 0
                (px, py) = data_coords[first_index]
                first_dot = data_dots[first_index]

                lines = VGroup()
                for i, (cx, cy) in enumerate(cluster_coords):
                    line = Line(
                        start=axes.coords_to_point(px, py),
                        end=axes.coords_to_point(cx, cy),
                        color=cluster_colors[i]
                    )
                    lines.add(line)

                self.play(
                    LaggedStart(*[Create(line) for line in lines], lag_ratio=0.1)
                )
                self.wait(0.5)

                dists = [
                    math.dist((px, py), (cx, cy))
                    for (cx, cy) in cluster_coords
                ]
                min_index = dists.index(min(dists))

                self.play(first_dot.animate.set_color(cluster_colors[min_index]))
                self.wait(0.5)

                self.play(*[FadeOut(line) for line in lines])
                self.wait(0.5)

            cluster_assignments = [[] for _ in range(K)]
            for i, (px, py) in enumerate(data_coords):
                dists = [
                    math.dist((px, py), (cx, cy))
                    for (cx, cy) in cluster_coords
                ]
                min_idx = dists.index(min(dists))
                cluster_assignments[min_idx].append(i)

            anims = []
            for cluster_idx, indices in enumerate(cluster_assignments):
                col = cluster_colors[cluster_idx]
                for i in indices:
                    anims.append(data_dots[i].animate.set_color(col))

            if anims:
                self.play(*anims, run_time=1)
                self.wait(0.5)

            new_cluster_coords = []
            total_shift = 0.0
            for cluster_idx, indices in enumerate(cluster_assignments):
                if len(indices) == 0:
                    new_cluster_coords.append(cluster_coords[cluster_idx])
                    continue

                mean_x = sum(data_coords[i][0] for i in indices) / len(indices)
                mean_y = sum(data_coords[i][1] for i in indices) / len(indices)

                old_cx, old_cy = cluster_coords[cluster_idx]
                shift = math.dist((old_cx, old_cy), (mean_x, mean_y))
                total_shift += shift

                new_cluster_coords.append((mean_x, mean_y))

            for idx, (newx, newy) in enumerate(new_cluster_coords):
                (oldx, oldy) = cluster_coords[idx]

                if abs(newx - oldx) < 1e-9 and abs(newy - oldy) < 1e-9:
                    continue

                if not special_animation_done and idx == special_cluster_idx:
                    assigned_points = cluster_assignments[idx]
                    connection_lines = VGroup()
                    for point_idx in assigned_points:
                        px, py = data_coords[point_idx]
                        line = Line(
                            start=axes.coords_to_point(px, py),
                            end=axes.coords_to_point(newx, newy),
                            stroke_width=2,
                            color=cluster_colors[idx]
                        )
                        connection_lines.add(line)

                    self.play(
                        LaggedStart(*[Create(line) for line in connection_lines], lag_ratio=0.02),
                        run_time=1
                    )
                    self.wait(0.5)

                    new_centroid = Dot(
                        axes.coords_to_point(newx, newy),
                        color=cluster_colors[idx],
                        radius=0.14
                    )
                    self.play(FadeIn(new_centroid))

                    self.play(FadeOut(connection_lines))
                    self.wait(0.5)

                    self.play(Transform(cluster_dots[idx], new_centroid))
                    self.play(FadeOut(new_centroid))
                    self.wait(0.3)
                else:
                    self.play(
                        cluster_dots[idx].animate.move_to(
                            axes.coords_to_point(newx, newy)
                        ),
                        run_time=1
                    )

            special_animation_done = True
            cluster_coords = new_cluster_coords

            if total_shift < THRESHOLD:
                break

        self.wait(1)
        self.play(
            FadeOut(data_dots),
            FadeOut(cluster_dots),
            FadeOut(axes),
        )
        self.wait()


class EuclideanDistance(Scene):
    def construct(self):
        # 1) Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)

        self.play(Create(axes))

        # 2) Define an unclassified point
        px, py = 4, 4
        unclassified_dot = Dot(
            axes.coords_to_point(px, py),
            color=WHITE,
            radius=0.12
        )

        # 3) Define 3 origins (cluster centers) each with distinct color
        origins_coords = [(2, 2), (6, 7), (8, 2)]
        origins_colors = [RED, GREEN, BLUE]
        origins_dots = VGroup()
        for (cx, cy), ccol in zip(origins_coords, origins_colors):
            dot = Dot(
                axes.coords_to_point(cx, cy),
                color=ccol,
                radius=0.12
            )
            origins_dots.add(dot)

        # Show the unclassified dot and origin dots
        self.play(FadeIn(unclassified_dot, shift=UP))
        self.play(LaggedStart(*[FadeIn(dot, shift=UP) for dot in origins_dots], lag_ratio=0.2))
        self.wait(1)

        # 4) Draw lines & labels
        lines_and_labels = VGroup()

        for (cx, cy), origin_dot, ccol in zip(origins_coords, origins_dots, origins_colors):
            # Create line
            line = Line(
                start=origin_dot.get_center(),
                end=unclassified_dot.get_center(),
                color=ccol
            )
            # Compute Euclidean distance in the axes coordinate space
            dist_val = math.dist((cx, cy), (px, py))

            # Create label
            label = MathTex(f"{dist_val:.2f}", color=ccol).scale(0.7)
            # Place label perpendicular (offset) to the line at its midpoint
            midpoint = line.get_midpoint()
            line_angle = line.get_angle()
            # We rotate UP by the line's angle to get a perpendicular offset
            offset_dir = rotate_vector(UP, line_angle)
            label.move_to(midpoint + offset_dir * 0.3)  # Adjust 0.3 as needed

            lines_and_labels.add(line, label)

        # Animate lines first, then labels
        lines = [m for m in lines_and_labels if isinstance(m, Line)]
        labels = [m for m in lines_and_labels if isinstance(m, MathTex)]
        self.play(*[Create(line) for line in lines], run_time=2)
        self.play(*[FadeIn(lbl) for lbl in labels], run_time=1)
        self.wait(2)

        # 5) Show the Euclidean distance formula at the top
        formula = Tex(
            r"Euklidovská vzdálenost: $d(\mathbf{x}, \mathbf{y}) = \sqrt{x^2 - y^2}$"
        ).scale(0.6).to_edge(UP)

        self.play(Write(formula))
        self.wait(3)

        # 6) Fade out everything
        all_mobjects = VGroup(axes, unclassified_dot, origins_dots, lines_and_labels, formula)
        self.play(FadeOut(all_mobjects))
        self.wait()


class Qcirc(Scene):
    def construct(self):
        svg = SVGMobject("./assets/qcirc.svg").scale(3)
        svg.move_to(ORIGIN)
        self.play(FadeIn(svg))
        self.wait(5)


class SwapTest(ThreeDScene):
    def construct(self):
        # Global parameters
        SCALE = 1
        RAD = 1
        AXIS_TEXT = 0.5
        CONE_HEIGHT = 0.2      # Height of the arrow head (cone)
        CONE_WIDTH = 0.05      # Base radius of the cone
        CYLINDER_RADIUS = 0.03 # Radius of the arrow shaft (cylinder)

        # 1) Set camera orientation.
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)
       # 2) Create 3D axes with shorter lengths.
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=8, y_length=8, z_length=5,
            stroke_width=2
        )
        self.add(axes)
        title_text = Text("Swap Test").scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_text)
        self.play(Write(title_text))

        
        # 3) Create a semi-transparent Bloch sphere centered at the origin.
        sphere = Surface(
            lambda u, v: np.array([
                SCALE * RAD * np.sin(u) * np.cos(v),
                SCALE * RAD * np.sin(u) * np.sin(v),
                SCALE * RAD * np.cos(u)
            ]),
            u_range=[0, PI],
            v_range=[0, TAU],
            resolution=(30, 30)
        )
        sphere.set_style(
            fill_color=BLUE, fill_opacity=0.1,
            stroke_color=BLUE_E, stroke_opacity=0.2
        )
        self.add(sphere)
        
        # 4) Add equatorial circle (xy-plane) centered at the origin.
        circle_xy = ParametricFunction(
            lambda t: np.array([
                SCALE * RAD * np.cos(t),
                SCALE * RAD * np.sin(t),
                0
            ]),
            t_range=[0, TAU],
            color=GREEN,
        )
        circle_xy.set_opacity(0.3)
        self.add(circle_xy)
        
        # 5) Add labels.
        ket0 = Text("|0>").scale(AXIS_TEXT * SCALE)
        ket0.rotate(PI/2, axis=RIGHT).rotate(PI/2, axis=OUT)
        ket0.move_to(axes.c2p(0, 0, 1.15))
        
        ket1 = Text("|1>").scale(AXIS_TEXT * SCALE)
        ket1.rotate(PI/2, axis=RIGHT).rotate(PI/2, axis=OUT)
        ket1.move_to(axes.c2p(0, 0, -1.15))
        
        label_x = Text("x").scale(AXIS_TEXT * SCALE)
        label_x.rotate(PI/2, axis=RIGHT).rotate(PI/2, axis=OUT)
        label_x.move_to(axes.c2p(1.15, 0, 0))
        
        label_y = Text("y").scale(AXIS_TEXT * SCALE)
        label_y.rotate(PI/2, axis=RIGHT).rotate(PI/2, axis=OUT)
        label_y.move_to(axes.c2p(0, 1.15, 0))
        
        self.add(ket0, ket1, label_x, label_y)
        
        # 6) Create the first qubit state arrow.
        # Arrow 1: defined by (theta1, phi1) initially along z.
        theta1 = 0.0   # along z
        phi1   = 0.0
        state_point1 = np.array([
            RAD * np.sin(theta1) * np.cos(phi1),
            RAD * np.sin(theta1) * np.sin(phi1),
            RAD * np.cos(theta1)
        ])
        arrow_line1 = Line(ORIGIN, state_point1, color=RED, stroke_width=2)
        arrow_head1 = Cone(
            base_radius=CONE_WIDTH,
            height=CONE_HEIGHT,
            direction=state_point1,
            fill_color=RED,
            fill_opacity=1,
            stroke_width=0
        )
        arrow_head1.shift(state_point1)
        arrow1 = VGroup(arrow_line1, arrow_head1)
        self.add(arrow1)
        
        # Apply H gate: Rotate arrow1 by PI/2 about Y-axis.
        self.play(FadeOut(title_text))
        title_text = Text("Aplikování H hradla").scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_text)
        self.play(Write(title_text))
        self.play(Rotate(arrow1, angle=PI/2, axis=Y_AXIS, about_point=ORIGIN))
        self.wait(5)
        
        # --- New Step: Rotate arrow further and highlight its path ---
        self.play(FadeOut(title_text))
        title_text = Text("Zadání Polárních Souřadnic").scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_text)
        self.play(Write(title_text))
        
        # Create a traced path following the tip of arrow1.
        traced_path = TracedPath(arrow_line1.get_end, stroke_color=RED, stroke_width=4)
        self.add(traced_path)
        # Further rotate arrow1 by PI/2 about Z-axis and then PI/3 about X-axis.
        self.play(Rotate(arrow1, angle=PI/2, axis=Z_AXIS, about_point=ORIGIN))
        self.play(Rotate(arrow1, angle=PI/3, axis=X_AXIS, about_point=ORIGIN))
        self.play(FadeOut(traced_path))
        self.wait(5)
        # ---------------------------------------------------------------
        
        # 7) Create the second qubit state arrow.
        # Arrow 2: defined by (theta2, phi2)
        theta2 = 1.2
        phi2   = 1.5
        state_point2 = np.array([
            RAD * np.sin(theta2) * np.cos(phi2),
            RAD * np.sin(theta2) * np.sin(phi2),
            RAD * np.cos(theta2)
        ])
        arrow_line2 = Line(ORIGIN, state_point2, color=BLUE, stroke_width=2)
        arrow_head2 = Cone(
            base_radius=CONE_WIDTH,
            height=CONE_HEIGHT,
            direction=state_point2,
            fill_color=BLUE,
            fill_opacity=1,
            stroke_width=0
        )
        arrow_head2.shift(state_point2)
        arrow2 = VGroup(arrow_line2, arrow_head2)
        self.play(FadeOut(title_text))
        title_text = Text("Přidání Středu").scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_text)
        self.play(Write(title_text))
        self.play(FadeIn(arrow2))
        self.wait(5)
        
        # 8) Animate a swap between the two qubit states via rotation.
        # Use the lines (first elements) to compute tip positions.
        v1 = arrow1[0].get_end()
        v2 = arrow2[0].get_end()
        axis_swap = np.cross(v1, v2)
        if np.linalg.norm(axis_swap) < 1e-6:
            axis_swap = np.array([0, 0, 1])
        else:
            axis_swap = axis_swap / np.linalg.norm(axis_swap)
        dot_val = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        dot_val = np.clip(dot_val, -1, 1)
        angle_swap = np.arccos(dot_val)
        
        self.play(FadeOut(title_text))
        title_text = Text("Výměna qubitů").scale(0.8).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title_text)
        self.play(Write(title_text))
        self.play(
            Rotate(arrow1, angle=angle_swap, axis=axis_swap, about_point=ORIGIN),
            Rotate(arrow2, angle=-angle_swap, axis=axis_swap, about_point=ORIGIN),
            run_time=2
        )
        self.wait(5)
        
        # 9) Fade out everything.
        self.play(
            FadeOut(sphere),
            FadeOut(axes),
            FadeOut(circle_xy),
            FadeOut(ket0),
            FadeOut(ket1),
            FadeOut(label_x),
            FadeOut(label_y),
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(title_text)
        )
        self.wait()

class LingebraExplanation(Scene):
    def construct(self):
        # Detailed sections with frames based on the LaTeX document
        states_dict = {
            "1. Skalární součin": [
                MathTex(r"\mathbf{u}, \mathbf{v} \in \mathbb{C}^n", font_size=44),
                MathTex(r"\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\dagger \mathbf{v} = \sum_{i=1}^{n} \overline{u_i} v_i", font_size=44),
                MathTex(r"\text{kde } \mathbf{u}^\dagger \text{ je Hermitovsky sdružená transpozice}", font_size=40),
                MathTex(r"\text{a } \overline{u_i} \text{ značí komplexně sdružené číslo}", font_size=40),
            ],
            "2. Stavy qubitu v bázových vektorech": [
                MathTex(r"|0\rangle = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, |1\rangle = \begin{bmatrix} 0 \\ 1 \end{bmatrix}", font_size=44),
                MathTex(r"|x\rangle = a |0\rangle + b |1\rangle = \begin{bmatrix} a \\ b \end{bmatrix}", font_size=44),
                MathTex(r"|y\rangle = c |0\rangle + d |1\rangle = \begin{bmatrix} c \\ d \end{bmatrix}", font_size=44),
            ],
            "3. Počáteční stav": [
                MathTex(r"|\psi\rangle = |x\rangle |y\rangle", font_size=44),
                MathTex(r"|0\rangle |\psi\rangle = |0\rangle |x\rangle |y\rangle", font_size=44),
            ],
            "4. První Hadamardovo hradlo": [
                MathTex(r"H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}", font_size=44),
                MathTex(r"H |0\rangle = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)", font_size=44),
                MathTex(r"\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) |\psi\rangle", font_size=44),
            ],
            "5. Operátor CSWAP (Fredkinovo hradlo)": [
                MathTex(r"\text{CSWAP: kontrolovaná výměna dvou qubitů}", font_size=40),
                MathTex(r"|0,A,B\rangle \rightarrow |0,A,B\rangle", font_size=40),
                MathTex(r"|1,A,B\rangle \rightarrow |1,B,A\rangle", font_size=40),
                MathTex(r"F = \begin{bmatrix} 1&0&0&0&0&0&0&0 \\ 0&1&0&0&0&0&0&0 \\ 0&0&1&0&0&0&0&0 \\ 0&0&0&1&0&0&0&0 \\ 0&0&0&0&1&0&0&0 \\ 0&0&0&0&0&0&1&0 \\ 0&0&0&0&0&1&0&0 \\ 0&0&0&0&0&0&0&1 \end{bmatrix}", font_size=32),
            ],
            "6. Tenzorové součiny stavů": [
                MathTex(r"\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) |x\rangle |y\rangle = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} \otimes \begin{bmatrix} a \\ b \end{bmatrix} \otimes \begin{bmatrix} c \\ d \end{bmatrix}", font_size=40),
                MathTex(r"\begin{bmatrix} 1 \\ 1 \end{bmatrix} \otimes \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} a \\ b \\ a \\ b \end{bmatrix}", font_size=40),
                MathTex(r"\begin{bmatrix} a \\ b \\ a \\ b \end{bmatrix} \otimes \begin{bmatrix} c \\ d \end{bmatrix} = \begin{bmatrix} ac \\ ad \\ bc \\ bd \\ ac \\ ad \\ bc \\ bd \end{bmatrix}", font_size=38),
            ],
            "7. Aplikace CSWAP hradla": [
                MathTex(r"F \begin{bmatrix} ac \\ ad \\ bc \\ bd \\ ac \\ ad \\ bc \\ bd \end{bmatrix} = \begin{bmatrix} ac \\ ad \\ bc \\ bd \\ ac \\ bc \\ ad \\ bd \end{bmatrix}", font_size=40),
                MathTex(r"= \frac{1}{\sqrt{2}}(|0\rangle |x\rangle |y\rangle + |1\rangle |y\rangle |x\rangle)", font_size=40),
                MathTex(r"= \frac{1}{\sqrt{2}}(|0\rangle |\psi\rangle + |1\rangle \mathcal{F}(|\psi\rangle))", font_size=40),
                MathTex(r"\text{kde } \mathcal{F}(|\psi\rangle) = |y\rangle |x\rangle", font_size=40),
            ],
            "8. Druhé Hadamardovo hradlo": [
                MathTex(r"H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)", font_size=44),
                MathTex(r"H \otimes I \otimes I [\frac{1}{\sqrt{2}}(|0\rangle |\psi\rangle + |1\rangle \mathcal{F}(|\psi\rangle))]", font_size=38),
                MathTex(r"= \frac{1}{2}[|0\rangle(|x\rangle|y\rangle + |y\rangle|x\rangle) + |1\rangle(|x\rangle|y\rangle - |y\rangle|x\rangle)]", font_size=38),
                MathTex(r"= \frac{1}{2}[|0\rangle(I+\mathcal{F})|\psi\rangle + |1\rangle(I-\mathcal{F})|\psi\rangle]", font_size=38),
            ],
            "9. Pravděpodobnost měření ancilly": [
                MathTex(r"P(|0\rangle) = \left\|\frac{1}{2}(|x\rangle|y\rangle + |y\rangle|x\rangle)\right\|^2", font_size=40),
                MathTex(r"= \frac{1}{4}(\||xy\rangle\|^2 + 2\langle xy|yx\rangle + \||yx\rangle\|^2)", font_size=40),
                MathTex(r"= \frac{1}{4}(1 + 2\langle xy|yx\rangle + 1)", font_size=40),
                MathTex(r"= \frac{1}{2} + \frac{1}{2}\langle xy|yx\rangle", font_size=40),
            ],
            "10. Skalární součin a kosinus úhlu": [
                MathTex(r"|xy\rangle|yx\rangle = ([a,b] \otimes [c,d])^\top ([c,d] \otimes [a,b])", font_size=38),
                MathTex(r"= [ac,ad,bc,bd] \begin{bmatrix} ca \\ cb \\ da \\ db \end{bmatrix}", font_size=38),
                MathTex(r"= (ac)^2 + 2abcd + (bd)^2", font_size=38),
                MathTex(r"(|x\rangle \cdot |y\rangle)^2 = (ac + bd)^2 = (ac)^2 + 2abcd + (bd)^2", font_size=38),
                MathTex(r"\Rightarrow |xy\rangle|yx\rangle = |\langle x|y\rangle|^2", font_size=40),
            ],
            "11. Vztah s kosinem úhlu": [
                MathTex(r"\langle x|y\rangle = \cos\vartheta", font_size=44),
                MathTex(r"P(|0\rangle) = \frac{1}{2} + \frac{1}{2}\cos^2\vartheta", font_size=44),
                MathTex(r"|\cos\vartheta| = \sqrt{2P(|0\rangle) - 1}", font_size=44),
            ],
            "12. Závěr": [
                MathTex(r"\text{Existuje korelace mezi:}", font_size=42),
                MathTex(r"\text{- pravděpodobností měření ancilly ve stavu }|0\rangle", font_size=38),
                MathTex(r"\text{- absolutní hodnotou kosinu úhlu mezi stavy }|x\rangle\text{ a }|y\rangle", font_size=38),
                MathTex(r"\text{To můžeme využít pro kvantovou verzi k-means}", font_size=40),
            ],
        }

        # Animate sections
        title = Tex("1. Skalární součin", font_size=48).to_edge(UP)
        self.play(Write(title), run_time=0.8)
        self.wait(0.5)

        for i, (section, frames) in enumerate(states_dict.items()):
            if i > 0:  # Skip first section title which is already displayed
                new_title = Tex(section, font_size=48).to_edge(UP)
                self.play(TransformMatchingTex(title, new_title, run_time=0.8))
                title = new_title
                self.wait(0.5)

            current = frames[0].move_to(ORIGIN)
            self.play(Write(current, run_time=0.8))
            self.wait(1.5)
            
            for frame in frames[1:]:
                next_frame = frame.move_to(ORIGIN)
                self.play(ReplacementTransform(current, next_frame, run_time=0.8))
                self.wait(1.5)
                current = next_frame
                
            self.play(FadeOut(current), run_time=0.5)
            self.wait(0.3)

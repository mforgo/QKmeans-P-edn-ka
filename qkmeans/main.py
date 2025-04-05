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
        title = Text("Kvantový Kmeans", font_size=64)
        # Subtitle
        subtitle = Text("Ing. Vít Nováček, PhD, Michal Forgó", font_size=20).next_to(title, DOWN, buff=0.4)
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
        random.seed(42)  # for reproducibility (remove to be fully random)
        K = random.randint(2, 4)  # random # of clusters in [2,4]

        # -----------------------------
        # 1) Setup axes
        # -----------------------------
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)

        self.play(Create(axes))

        # -----------------------------
        # 2) Generate data points
        # -----------------------------
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
            LaggedStart(
                *[FadeIn(dot, scale=0.5) for dot in data_dots],
                lag_ratio=0.1
            )
        )
        self.wait(1)

        # -----------------------------
        # 3) Generate cluster centers
        # -----------------------------
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
            LaggedStart(
                *[FadeIn(dot, scale=0.8) for dot in cluster_dots],
                lag_ratio=0.1
            )
        )
        self.wait(1)

        # -----------------------------
        # 4) K-Means Iteration
        # -----------------------------
        MAX_ITER = 10
        THRESHOLD = 0.02  # if cluster centers move less than this, we consider stable

        for iteration in range(MAX_ITER):
            # a) Demonstrate lines to the FIRST data point ONLY in iteration 0
            if iteration == 0:
                first_index = 0
                (px, py) = data_coords[first_index]
                first_dot = data_dots[first_index]

                # Draw lines from the first point to each cluster center
                lines = VGroup()
                for i, (cx, cy) in enumerate(cluster_coords):
                    line = Line(
                        start=axes.coords_to_point(px, py),
                        end=axes.coords_to_point(cx, cy),
                        color=cluster_colors[i]
                    )
                    lines.add(line)

                # Animate creation of lines
                self.play(
                    LaggedStart(*[Create(line) for line in lines], lag_ratio=0.1)
                )
                self.wait(0.5)

                # Find nearest center to the first point
                dists = [
                    math.dist((px, py), (cx, cy))
                    for (cx, cy) in cluster_coords
                ]
                min_index = dists.index(min(dists))
                # Color the first dot
                self.play(first_dot.animate.set_color(cluster_colors[min_index]))
                self.wait(0.5)

                # Fade out lines
                self.play(*[FadeOut(line) for line in lines])
                self.wait(0.5)

            # b) Assign each point to its nearest cluster
            cluster_assignments = [[] for _ in range(K)]
            for i, (px, py) in enumerate(data_coords):
                dists = [
                    math.dist((px, py), (cx, cy))
                    for (cx, cy) in cluster_coords
                ]
                min_idx = dists.index(min(dists))
                cluster_assignments[min_idx].append(i)

            # c) Animate coloring of the points
            anims = []
            for cluster_idx, indices in enumerate(cluster_assignments):
                col = cluster_colors[cluster_idx]
                for i in indices:
                    anims.append(data_dots[i].animate.set_color(col))

            # If there's at least one coloring animation, play them
            if anims:
                self.play(*anims, run_time=1)
                self.wait(0.5)

            # d) Recompute each cluster center as the mean of assigned points
            new_cluster_coords = []
            total_shift = 0.0
            for cluster_idx, indices in enumerate(cluster_assignments):
                if len(indices) == 0:
                    # No points assigned => keep old position
                    new_cluster_coords.append(cluster_coords[cluster_idx])
                    continue

                mean_x = sum(data_coords[i][0] for i in indices) / len(indices)
                mean_y = sum(data_coords[i][1] for i in indices) / len(indices)

                old_cx, old_cy = cluster_coords[cluster_idx]
                shift = math.dist((old_cx, old_cy), (mean_x, mean_y))
                total_shift += shift

                new_cluster_coords.append((mean_x, mean_y))

            # e) Animate movement of cluster centers
            center_animations = []
            for idx, (newx, newy) in enumerate(new_cluster_coords):
                (oldx, oldy) = cluster_coords[idx]
                if abs(newx - oldx) > 1e-9 or abs(newy - oldy) > 1e-9:
                    center_animations.append(
                        cluster_dots[idx].animate.move_to(
                            axes.coords_to_point(newx, newy)
                        )
                    )

            if center_animations:
                self.play(*center_animations, run_time=1)
                self.wait(0.5)

            # Update cluster positions
            cluster_coords = new_cluster_coords

            # f) Check for convergence
            if total_shift < THRESHOLD:
                break

        # Final pause, fade out everything
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
            r"Euclidean Distance: $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_i (x_i - y_i)^2}$"
        ).scale(0.6).to_edge(UP)

        self.play(Write(formula))
        self.wait(3)

        # 6) Fade out everything
        all_mobjects = VGroup(axes, unclassified_dot, origins_dots, lines_and_labels, formula)
        self.play(FadeOut(all_mobjects))
        self.wait()

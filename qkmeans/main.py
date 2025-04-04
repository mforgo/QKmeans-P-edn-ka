
from manim import *
import random
import numpy as np

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
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],  # from 0 to 10 with tick spacing of 1
            y_range=[0, 10, 1],  # from 0 to 10 with tick spacing of 1
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)  # Shift slightly down for aesthetics

        # Animate the axes on screen
        self.play(Create(axes))

        # Generate 20 random points and place them on the axes
        dots = VGroup()
        for _ in range(20):
            # Random x, y in [0, 9]
            x = random.uniform(0, 9)
            y = random.uniform(0, 9)
            dot = Dot(
                point=axes.coords_to_point(x, y),
                color=BLUE
            )
            dots.add(dot)

        # Animate dots appearing one by one (with a small lag)
        self.play(LaggedStart(*[FadeIn(dot, scale=0.5) for dot in dots], lag_ratio=0.1))
        self.wait(20)

class Correlation(Scene):
    def construct(self):
        pass

class Qkmeans(Scene):
    def construct(self):
        pass

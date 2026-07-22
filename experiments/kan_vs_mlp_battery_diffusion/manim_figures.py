"""
manim_figures.py -- Professional, equation-free animations of the Phase 3
KAN vs MLP comparison, for a non-academic audience.

Reads plain numpy/json data only (outputs/phase3_results.json and
outputs/manim_data/solution_fit_dense.npz) -- no torch import here, so
this can run in a separate, lighter virtual environment than the one
used to train the models.

Three scenes, one per empirical finding:
  TrainingRace      -- both models' prediction error falling during training
  CostVsAccuracy    -- training time vs. final error, 5 runs each
  RealityCheck      -- predicted vs. real battery-charging curve, over time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import *

NEAR_BLACK = "#0e0e0e"
OFFWHITE = "#f5f0e8"
MLP_BLUE = "#4fa8ff"
KAN_ORANGE = "#ff5300"
GREY_TEXT = "#9a9a9a"

DATA_DIR = Path(__file__).parent / "outputs"
with open(DATA_DIR / "phase3_results.json") as f:
    PAYLOAD = json.load(f)


def curve_arrays(payload, arch, key):
    runs = payload["results"][arch]
    epochs = np.array([h["epoch"] for h in runs[0]["history"]])
    vals = np.array([[h[key] for h in r["history"]] for r in runs])
    return epochs, vals.mean(axis=0)


class TrainingRace(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        title = Text("Learning to Solve the Physics", font_size=36, color=OFFWHITE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        subtitle = Text("Prediction error while training, averaged over 5 runs",
                        font_size=20, color=GREY_TEXT).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle), run_time=1.0)

        epochs, mlp_mean = curve_arrays(PAYLOAD, "mlp", "rel_l2")
        _, kan_mean = curve_arrays(PAYLOAD, "kan", "rel_l2")
        log_mlp, log_kan = np.log10(mlp_mean), np.log10(kan_mean)
        y_all = np.concatenate([log_mlp, log_kan])
        y_floor = y_all.min() - 0.15
        y_span = (y_all.max() + 0.15) - y_floor
        # Shift so the plotted range starts at 0 -- keeps Axes' default
        # x-axis (drawn at y=0) pinned to the BOTTOM of the chart instead
        # of floating wherever raw log10(error)=0 happens to fall.
        shifted_mlp = log_mlp - y_floor
        shifted_kan = log_kan - y_floor

        axes = Axes(
            x_range=[0, epochs.max(), epochs.max() / 6],
            y_range=[0, y_span, y_span / 5],
            x_length=10, y_length=5,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.35)

        x_label = Text("Training Progress", font_size=20, color=OFFWHITE).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Prediction error", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        mlp_points = [axes.c2p(e, l) for e, l in zip(epochs, shifted_mlp)]
        kan_points = [axes.c2p(e, l) for e, l in zip(epochs, shifted_kan)]
        mlp_curve = VMobject(color=MLP_BLUE, stroke_width=5).set_points_smoothly(mlp_points)
        kan_curve = VMobject(color=KAN_ORANGE, stroke_width=5).set_points_smoothly(kan_points)
        mlp_dot = Dot(color=MLP_BLUE, radius=0.09).move_to(mlp_points[0])
        kan_dot = Dot(color=KAN_ORANGE, radius=0.09).move_to(kan_points[0])

        self.play(
            Create(mlp_curve), Create(kan_curve),
            MoveAlongPath(mlp_dot, mlp_curve), MoveAlongPath(kan_dot, kan_curve),
            run_time=4.5, rate_func=linear,
        )

        mlp_label = Text("MLP", font_size=26, color=MLP_BLUE, weight=BOLD).next_to(mlp_points[-1], UP, buff=0.25)
        kan_label = Text("KAN", font_size=26, color=KAN_ORANGE, weight=BOLD).next_to(kan_points[-1], DOWN, buff=0.3)
        self.play(FadeIn(mlp_label, shift=DOWN * 0.1), FadeIn(kan_label, shift=UP * 0.1), run_time=0.6)
        self.wait(1.8)


class CostVsAccuracy(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        title = Text("Prediction Error vs Training Time (5 runs each)",
                    font_size=30, color=OFFWHITE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=1.0)

        mlp_runs, kan_runs = PAYLOAD["results"]["mlp"], PAYLOAD["results"]["kan"]
        mlp_times = np.array([r["wall_time"] for r in mlp_runs])
        mlp_errs = np.log10(np.array([r["final_rel_l2"] for r in mlp_runs]))
        kan_times = np.array([r["wall_time"] for r in kan_runs])
        kan_errs = np.log10(np.array([r["final_rel_l2"] for r in kan_runs]))

        x_max = max(mlp_times.max(), kan_times.max()) * 1.15
        y_all = np.concatenate([mlp_errs, kan_errs])
        y_floor = y_all.min() - 0.15
        y_span = (y_all.max() + 0.15) - y_floor
        shifted_mlp_errs = mlp_errs - y_floor
        shifted_kan_errs = kan_errs - y_floor

        axes = Axes(
            x_range=[0, x_max, x_max / 5],
            y_range=[0, y_span, y_span / 4],
            x_length=10, y_length=5,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.05)

        x_label = Text("Training Time [s]", font_size=20, color=OFFWHITE)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Prediction error", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        mlp_dots = VGroup(*[Dot(axes.c2p(tm, e), color=MLP_BLUE, radius=0.1)
                            for tm, e in zip(mlp_times, shifted_mlp_errs)])
        kan_dots = VGroup(*[Dot(axes.c2p(tm, e), color=KAN_ORANGE, radius=0.1)
                            for tm, e in zip(kan_times, shifted_kan_errs)])

        self.play(LaggedStart(*[GrowFromCenter(d) for d in mlp_dots], lag_ratio=0.2), run_time=1.2)
        mlp_tag = Text("MLP", font_size=24, color=MLP_BLUE, weight=BOLD).next_to(mlp_dots, UP, buff=0.3)
        self.play(FadeIn(mlp_tag, shift=DOWN * 0.1), run_time=0.5)

        self.play(LaggedStart(*[GrowFromCenter(d) for d in kan_dots], lag_ratio=0.2), run_time=1.2)
        kan_tag = Text("KAN", font_size=24, color=KAN_ORANGE, weight=BOLD).next_to(kan_dots, UP, buff=0.3)
        self.play(FadeIn(kan_tag, shift=DOWN * 0.1), run_time=0.5)

        speedup = kan_times.mean() / mlp_times.mean()
        caption = Text(f"KAN: more accurate, ~{speedup:.0f}× slower to train",
                       font_size=24, color=OFFWHITE).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.6)
        self.wait(2.0)


class RealityCheck(Scene):
    def construct(self):
        self.camera.background_color = NEAR_BLACK

        data = np.load(DATA_DIR / "manim_data" / "solution_fit_dense.npz")
        x_um, t_min = data["x_um"], data["t_min"]
        charge_ref, charge_mlp, charge_kan = data["charge_ref"], data["charge_mlp"], data["charge_kan"]
        n_frames = len(t_min)

        title = Text("Does It Match Reality?", font_size=36, color=OFFWHITE, weight=BOLD).to_edge(UP, buff=0.5)
        subtitle = Text("Charging a battery electrode: simulated (ground truth) vs. predicted",
                        font_size=19, color=GREY_TEXT).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title, shift=DOWN * 0.2), FadeIn(subtitle), run_time=1.0)

        axes = Axes(
            x_range=[x_um.min(), x_um.max(), (x_um.max() - x_um.min()) / 4],
            y_range=[0, 100, 25],
            x_length=10, y_length=4.3,
            axis_config={"color": OFFWHITE, "stroke_width": 2, "include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.6)

        left_label = Text("Charging surface", font_size=17, color=GREY_TEXT)
        left_label.next_to(axes.c2p(x_um.min(), 0), DOWN, buff=0.3)
        right_label = Text("Particle core", font_size=17, color=GREY_TEXT)
        right_label.next_to(axes.c2p(x_um.max(), 0), DOWN, buff=0.3)
        y_label = Text("Charge level", font_size=20, color=OFFWHITE).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.25)

        self.play(Create(axes), FadeIn(left_label), FadeIn(right_label), FadeIn(y_label), run_time=1.0)

        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=OFFWHITE, stroke_width=6),
                   Text("Real physics", font_size=16, color=OFFWHITE)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=MLP_BLUE, stroke_width=4),
                   Text("MLP prediction", font_size=16, color=MLP_BLUE)).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=KAN_ORANGE, stroke_width=4),
                   Text("KAN prediction", font_size=16, color=KAN_ORANGE)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.7).shift(DOWN * 0.9)
        self.play(FadeIn(legend), run_time=0.8)

        time_tracker = ValueTracker(0.0)

        def interp(arr):
            idx_f = time_tracker.get_value() * (n_frames - 1)
            i0 = int(np.floor(idx_f))
            i1 = min(i0 + 1, n_frames - 1)
            w = idx_f - i0
            return arr[i0] * (1 - w) + arr[i1] * w

        def make_curve(y_values, color, width, opacity=1.0):
            points = [axes.c2p(xv, yv) for xv, yv in zip(x_um, y_values)]
            return VMobject(stroke_color=color, stroke_width=width,
                            stroke_opacity=opacity).set_points_smoothly(points)

        # The three curves nearly coincide almost everywhere, so a KAN line
        # drawn at the same width would completely hide MLP and the
        # reference underneath it. Nesting decreasing widths (thick pale
        # reference -> medium MLP -> thin KAN) keeps all three visible as
        # concentric "halo" lines, and any real deviation between them
        # shows up as one line straying outside another's band.
        ref_curve = always_redraw(lambda: make_curve(interp(charge_ref), OFFWHITE, 10, opacity=0.9))
        mlp_curve = always_redraw(lambda: make_curve(interp(charge_mlp), MLP_BLUE, 5))
        kan_curve = always_redraw(lambda: make_curve(interp(charge_kan), KAN_ORANGE, 2))
        time_label = always_redraw(lambda: Text(
            f"t = {interp(t_min):.1f} min", font_size=24, color=OFFWHITE,
        ).to_corner(UR, buff=0.6))

        self.add(ref_curve, mlp_curve, kan_curve, time_label)
        self.play(FadeIn(VGroup(ref_curve, mlp_curve, kan_curve, time_label)), run_time=0.6)
        self.play(time_tracker.animate.set_value(1.0), run_time=8.0, rate_func=linear)
        self.wait(0.4)

        caption = Text("Both models learned to reproduce the real physics.",
                       font_size=23, color=OFFWHITE).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=0.7)
        self.wait(1.8)

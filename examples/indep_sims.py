"""
Independence Simulations
===========================

Independence simulations are found in :mod:`hyppo.tools`. Here, we visualize what these
simulations look like. Noise-free simulations are overlaid over the noisy simulation.
Note that the last 2 simulations have no noise parameter. These simulations were chosen
as an aggregate of many popularly tested equations in the literature.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyppo.tools import (
    circle,
    cubic,
    diamond,
    ellipse,
    exponential,
    fourth_root,
    joint_normal,
    linear,
    logarithmic,
    multimodal_independence,
    multiplicative_noise,
    quadratic,
    sin_four_pi,
    sin_sixteen_pi,
    spiral,
    square,
    step,
    two_parabolas,
    uncorrelated_bernoulli,
    w_shaped,
)

# make plots look pretty
sns.set(color_codes=True, style="white", context="talk", font_scale=2)

# constants
NOISY = 100  # sample size of noisy simulation
NO_NOISE = 1000  # sample size of noise-free simulation

# dictionary mapping of simulations
SIMULATIONS = [
    (linear, "Linear"),
    (exponential, "Exponential"),
    (cubic, "Cubic"),
    (joint_normal, "Joint Normal"),
    (step, "Step"),
    (quadratic, "Quadratic"),
    (w_shaped, "W-Shaped"),
    (spiral, "Spiral"),
    (uncorrelated_bernoulli, "Bernoulli"),
    (logarithmic, "Logarithmic"),
    (fourth_root, "Fourth Root"),
    (sin_four_pi, "Sine 4\u03C0"),
    (sin_sixteen_pi, "Sine 16\u03C0"),
    (square, "Square"),
    (two_parabolas, "Two Parabolas"),
    (circle, "Circle"),
    (ellipse, "Ellipse"),
    (diamond, "Diamond"),
    (multiplicative_noise, "Noise"),
    (multimodal_independence, "Independence"),
]


# make a function that runs the code depending on the simulation
def plot_sims():
    """Plot simulations"""
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(28, 24))

    plt.suptitle("Independence Simulations", y=0.93, va="baseline")

    count = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            count = 5 * i + j
            sim, sim_title = SIMULATIONS[count]

            # the multiplicative noise and independence simulation don't have a noise
            # parameter
            if sim_title in ["Noise", "Independence"]:
                x, y = sim(NO_NOISE, 1)
                x_no_noise, y_no_noise = x, y
            else:
                x, y = sim(NOISY, 1, noise=True)
                x_no_noise, y_no_noise = sim(NO_NOISE, 1)

            # plot the nose and noise-free sims
            col.scatter(x, y, c="#d9d9d9", marker="+", label="Noisy")
            col.scatter(
                x_no_noise, y_no_noise, c="#525252", marker="x", label="No Noise"
            )

            # make the plot look pretty
            col.set_title("{}".format(sim_title))
            col.set_xticks([])
            col.set_yticks([])
            if count == 16:
                col.set_ylim([-1, 1])
            sns.despine(left=True, bottom=True, right=True)

    leg = plt.legend(
        bbox_to_anchor=(0.5, 0.1),
        bbox_transform=plt.gcf().transFigure,
        ncol=5,
        loc="upper center",
    )
    leg.get_frame().set_linewidth(0.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.subplots_adjust(hspace=0.75)


# run the created function for the simultions
plot_sims()

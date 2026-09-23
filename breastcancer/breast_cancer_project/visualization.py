"""
Visualization functions.
"""

import matplotlib.pyplot as plt


def plot_class_distribution(class_distribution):
    """Plot the class distribution as a bar chart."""
    class_distribution.plot(
        x="Class",
        y="Count",
        kind="bar",
        legend=False,
        color=["lightyellow", "pink"],
    )

    plt.ylabel("number of observations")
    plt.title("class distribution")
    plt.xticks(rotation=0)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import colorbar
import pandas as pd

# set default configuration
print(plt.style.available)


def viz_missing(df: pd.DataFrame, fig_size: tuple[int, int] = (12, 6)) -> None:
    """
    Visualize missing data percentage and heatmap.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to visualize.
        fig_size (tuple[int, int]): figure size to use.

    Returns:
        None
    """
    # Select missing data
    na_df = df.isna()

    # Calculate the missing data percentage
    na_percent: pd.DataFrame = 100 * na_df.mean()
    # Create a figure and two subplots
    fig, (ax_bar, ax_heat) = plt.subplots(2, 1, figsize=fig_size, sharex=False)
    # region: Plot the missing data percentage on the first subplot
    na_percent.plot(
        kind="bar", ax=ax_bar, fontsize=6, color="black", ylim=[0, 100], width=0.5
    )
    ax_bar.grid(False, which="both", axis="both")
    # ax_bar.set_xticks([])
    ax_bar.tick_params(axis="both", which="both", labelsize=6, length=2)
    ax_bar.set_ylabel("% of Missing Values")
    ax_bar.set_title("Missing Values Analysis")
    # endregion
    # region Plot the missing data heatmap on the second subplot
    sns.heatmap(na_df, cmap="binary", cbar=False, square=False, ax=ax_heat)
    ax_heat.tick_params(axis="both", which="major", labelsize=6, length=0)
    ax_heat.set_xlabel("Features")
    ax_heat.set_ylabel("Index")
    # endregion
    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.1)
    # Display the plot
    plt.show()


def viz_missing_interactive(
    df: pd.DataFrame, fig_size: tuple[int, int] = (1280, 500), return_fig=False
) -> tuple[px.bar, px.imshow] | None:
    """
    Visualize missing data percentage and heatmap.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to visualize.
    """
    # Calculate the missing data percentage
    na_percent = df.isna().mean() * 100

    # region: Create a bar chart of the missing data percentage
    fig_bar = px.bar(na_percent, title="Missing Data Percentage", orientation="v")
    fig_bar.update_layout(
        xaxis_title="Features",
        yaxis_title="% of Missing Values",
        yaxis_range=[0, 100],
        width=fig_size[0],
        height=fig_size[1],
    )
    fig_bar.update_traces(marker_color="black", showlegend=False)
    # endregion

    # region Create a heatmap of the missing data
    fig_heat = px.imshow(df.isna(), color_continuous_scale=["white", "black"])
    fig_heat.update_layout(
        title="Missing Data Heatmap",
        xaxis_title="Features",
        yaxis_title="Index",
        coloraxis_showscale=False,
        width=fig_size[0],
        height=fig_size[1],
    )
    # endregion

    if return_fig:
        return fig_bar, fig_heat
    else:
        # Display the plots
        fig_bar.show()
        fig_heat.show()
        return None


# create a figure with two subplots
def viz_distribution(
    data: pd.DataFrame,
    x_target: str,
    fig_size=(12, 6),
    use_density: bool = False,
    cumulate=False,
):
    """Display distribution using histogram and boxplot

    Args:
        data (pd.DataFrame): data frame to vizualize distrubitions
        x_target (str): _description_
        use_density (bool, optional): _description_. Defaults to False.
        cumulate (bool, optional): _description_. Defaults to False.
    """
    sns.set_style("dark")  # darkgrid, whitegrid, dark, white, ticks)
    fig, (ax_hist, ax_box) = plt.subplots(
        figsize=fig_size,
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.9, 0.3)},
    )

    # region: Plot & annotate mean line and median line
    x_mean = data[x_target].mean()
    x_median = data[x_target].median()
    ax_hist.axvline(
        x_mean,
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax_hist.axvline(
        x_median,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    # Add annotations for mean and median
    ax_hist.annotate(
        "Mean",
        xy=(x_mean, 0),
        xytext=(x_mean + 0.1, 0),
        arrowprops=dict(
            facecolor="black",
            arrowstyle="fancy",
        ),
    )

    ax_hist.annotate(
        "Median",
        xy=(x_median, 50),
        xytext=(x_median + 0.1, 50),
        arrowprops=dict(
            facecolor="black",
            arrowstyle="fancy",
        ),
    )
    # endregion

    sns.histplot(
        data,
        x=x_target,
        bins="auto",
        ax=ax_hist,
        cumulative=cumulate,
        kde=use_density,
    )
    # alernative : data.hist(column=x_target, ax=ax_hist, bins="auto", density=True)

    # plot the boxplot on the top subplot
    sns.boxplot(data, x=x_target, ax=ax_box, orient="h")
    ax_box.set(xlabel=x_target)

    # add some labels and titles
    ax_hist.set(xlabel=x_target, ylabel="Frequency")
    ax_box.set(ylabel="")
    plt.show()


def viz_correlations(
    data: pd.DataFrame, corr_method: str = "pearson", cutoff=0.0, fig_size=9
):
    corr_matrix = data.corr(numeric_only=True, method=corr_method)  # type: ignore
    # drop all features with no correlation values
    corr_matrix = corr_matrix.dropna(how="all", axis="columns").dropna(
        how="all",
        axis="index",
    )
    mask = corr_matrix.abs() >= cutoff
    plt.figure(figsize=(fig_size, fig_size))
    sns.set_style("white")
    ax = sns.heatmap(
        data=corr_matrix,
        mask=~mask,
        annot=True,
        fmt=".2f",
        square=True,
        vmin=-1,
        linewidths=0.1,
        cmap="RdYlGn",
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Correlation Matrix")
    ax.set_xlabel("")
    ax.set_ylabel("")


def viz_clusters_correlations(
    data: pd.DataFrame, corr_method="pearson", cutoff=0.0, fig_size=(7, 7)
):
    corr_matrix = data.corr(numeric_only=True, method=corr_method)
    # drop all features with no correlation values
    corr_matrix = corr_matrix.dropna(how="all", axis="columns").dropna(
        how="all", axis="index"
    )
    mask = corr_matrix.abs() >= cutoff
    plt.figure(figsize=fig_size)
    sns.set_style("white")
    ax = sns.clustermap(
        data=corr_matrix,
        mask=~mask,
        method="average",
        annot=True,
        fmt=".2f",
        vmin=-1,
        linewidths=0.05,
        cmap="RdYlGn",
        cbar_kws={"shrink": 0.8},
    )


def viz_correlation_interactive(
    data, corr_method="pearson", cutoff=0.0, fig_size=(280, 280)
):
    corr_matrix = data.corr(numeric_only=True, method=corr_method)
    corr_matrix = corr_matrix.dropna(how="all", axis="columns").dropna(
        how="all", axis="index"
    )
    mask = corr_matrix.abs() >= cutoff
    fig_heat = px.imshow(corr_matrix, color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
    fig_heat.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
        width=fig_size[0],
        height=fig_size[1],
    )
    fig_heat.update_traces(showscale=True, colorbar=dict(title="Correlation"))
    fig_heat.update_xaxes(side="top")

    fig_heat.show()


def viz_scatter(df: pd.DataFrame):
    raise NotImplementedError

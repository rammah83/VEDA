

# set default configuration
print(plt.style.available)
plt.style.use("ggplot")


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
    na_percent: pd.DataFrame = na_df.mean() * 100

    # Create a figure and two subplots
    fig, (ax_bar, ax_heat) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # region: Plot the missing data percentage on the first subplot
    na_percent.plot(kind="bar", ax=ax_bar, fontsize=6, color="black", ylim=[0, 100])
    ax_bar.grid(False, which="both", axis="both")
    ax_bar.tick_params(axis="both", which="both", labelsize=6, length=0)
    ax_bar.set_ylabel("% of Missing Values")
    ax_bar.set_title("Missing Values Analysis")
    # endregion

    # region Plot the missing data heatmap on the second subplot
    sns.heatmap(na_df, cmap="binary", cbar=False, square=False, ax=ax_heat)
    ax_heat.tick_params(axis="both", which="major", labelsize=6, length=0)
    ax_heat.set_xlabel("Features")
    # endregion

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.1)

    # Display the plot
    plt.show()


from matplotlib import colorbar
import plotly.express as px


def viz_missing_interactive(
    df: pd.DataFrame, fig_size: tuple[int, int] = (1280, 500), return_fig=False
) -> tuple[px.bar, px.imshow]:
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
        coloraxis_showscale=False,
        width=fig_size[0],
        height=fig_size[1],
    )
    # endregion

    if return_fig:
        return fig_bar, fig_heat
    else:
        # Display the plots
        fig_heat.show()
        fig_bar.show()
        
        
# create a figure with two subplots
def viz_distribution(data: pd.DataFrame, x_target: str, use_density: bool = False):
    sns.set_style("dark")  # darkgrid, whitegrid, dark, white, ticks)
    fig, (ax_hist, ax_box) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.8, 0.2)}
    )

    sns.histplot(
        data,
        x=x_target,
        ax=ax_hist,
        kde=use_density,
    )

    # plot the boxplot on the top subplot
    sns.boxplot(data, x=x_target, ax=ax_box, orient="h")
    ax_box.set(xlabel=x_target)

    # add some labels and titles
    ax_hist.set(xlabel=x_target, ylabel="Frequency")
    ax_box.set(ylabel="")
    plt.show()



def viz_correlations(data: pd.DataFrame, corr_method: str = "pearson", cutoff=0.0):
    corr_matrix = data.corr(numeric_only=True, method=corr_method)  # type: ignore
    # drop all features with no correlation values
    corr_matrix = corr_matrix.dropna(how="all", axis="columns").dropna(
        how="all", axis="index"
    )
    mask = corr_matrix.abs() >= cutoff
    plt.figure(figsize=(9, 9))
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
        annot_kws={'fontsize':8},
    )
    ax.set_title("Correlation Matrix")
    ax.set_xlabel("")
    ax.set_ylabel("")


def viz_clusters_correlations(data: pd.DataFrame, corr_method="pearson", cutoff=0.0):
    corr_matrix = data.corr(numeric_only=True, method=corr_method)
    # drop all features with no correlation values
    corr_matrix = corr_matrix.dropna(how="all", axis="columns").dropna(
        how="all", axis="index"
    )
    mask = corr_matrix.abs() >= cutoff
    plt.figure(figsize=(9, 9))
    sns.set_style("white")
    ax = sns.clustermap(
        data=corr_matrix,
        mask=~mask,
        method="average",
        annot=True,
        fmt=".2f",
        vmin=-1,
        linewidths=0.1,
        cmap="RdYlGn",
        cbar_kws={"shrink": 0.8},
    )

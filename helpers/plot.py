
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_high_quality_mpl_params():

    """
    Setup plotting parameters.
    
    Setup environment to make 
    fancy looking plots.
    Inspiration from Chris Ketter.
    Good quality for exporting.

    Side Effects
    ------------
    - Changes matplotlib rc parameters.
    """

    mpl.rcParams["figure.figsize"] = (6, 4)
    mpl.rcParams["figure.dpi"] = 400
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["figure.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["figure.labelsize"] = 30
    mpl.rcParams["xtick.labelsize"] = 12 
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titley"] = None
    mpl.rcParams["axes.titlepad"] = 4
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.framealpha"] = 0
    mpl.rcParams["legend.markerscale"] = 1
    mpl.rcParams["legend.fontsize"] = 7.5


def add_plot_note(ax, text:str, fontsize="medium"):

    """
    Annotate a plot in the upper right corner,
    above the plot box.

    This doesn't work for 3D plots.
    """
    
    ax.text(
        -0.15,
        1.05, 
        text, 
        horizontalalignment="left", 
        verticalalignment="bottom", 
        transform=ax.transAxes, 
        fontsize=fontsize
    )


def save_model_evaluation_plot(type, model_settings, dataset_settings, path_to_plots_dir):
    
    file_name = f"{model_settings.name}_{dataset_settings.common.level}_{dataset_settings.set.num_events_per_set}_{type}.png"
    path = Path(path_to_plots_dir).joinpath(file_name)
    plt.savefig(path, bbox_inches="tight")


def to_x10_notation(number):

    if number == 0:
        return 0
    
    string = f"{number : .1e}"
    e_index = string.find('e')
    string = r"$" + string[:e_index] + r"\times 10^{" + str(int(string[e_index+1:])) + r"}$"
    return string


def stats_legend(
    data_array,
    extra_description=None,
    show_mean=True,
    show_count=True,
    show_stdev=True,
):
    
    """
    Make a legend label similar to the roots stats box.

    Return a string meant to be used as a label for a matplotlib plot.
    Displayable stats are mean, count, and standard deviation.
    """
    
    def calculate_stats(data):
        mean_ = mean(data)
        count = len(data)
        stdev = std(data)
        stats = {
            "mean": mean_,
            "count": count,
            "stdev": stdev,
        }
        return stats
    
    stats = calculate_stats(data_array)

    legend = ""
    if extra_description is not None:
        legend += r"\textbf{" + f"{extra_description}" + "}"
    if show_count:
        legend += f"\nCount: {to_x10_notation(stats['count'])}"
    if show_mean:
        legend += f"\nMean: {to_x10_notation(stats['mean'])}"
    if show_stdev:
        legend += f"\nStdev.: {to_x10_notation(stats['stdev'])}"

    return legend


def plot_afb(afb_results, ax, cmap, norm, alpha):
    
    for dc9, bin_mids, afbs, afb_errs in zip(*afb_results):
        
        is_standard_model = (dc9 == 0)

        color = cmap(norm(dc9), alpha=alpha) if not is_standard_model else "dimgrey"
        label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
        zorder = None if not is_standard_model else 10
        
        ax.scatter(
            bin_mids, 
            afbs, 
            label=label,
            color=color, 
            edgecolors='none',
            s=10,
            zorder=zorder,
        )

        ax.errorbar(
            bin_mids, 
            afbs, 
            yerr=afb_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
            zorder=zorder,
        )

    ax.set_xlabel(r"$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$A_{FB}$")
    ax.set_ylim(-0.25, 0.46)
    ax.legend()


def plot_s5(s5_results, ax, cmap, norm, alpha):

    for dc9, bin_mids, s5s, s5_errs in zip(*s5_results):
        
        is_standard_model = (dc9 == 0)

        color = cmap(norm(dc9), alpha=alpha) if not is_standard_model else "dimgrey"
        label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
        zorder = None if not is_standard_model else 10

        ax.scatter(
            bin_mids, 
            s5s, 
            label=label,
            color=color, 
            edgecolors='none',
            s=10,
            zorder=zorder,
        )

        ax.errorbar(
            bin_mids, 
            s5s, 
            yerr=s5_errs, 
            fmt='none', 
            ecolor=color, 
            elinewidth=0.5, 
            capsize=0, 
            zorder=zorder,
        )
    
    ax.set_xlabel(r"$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$S_{5}$")
    ax.set_ylim(-0.48, 0.38)
    ax.legend()


def plot_afb_and_s5(fig, axs_2x1, afb_results, s5_results, alpha=0.85):

    def get_colorbar_halfrange(afb_results, s5_results):
        
        assert min(afb_results.delta_C9_values) == min(s5_results.delta_C9_values)
        return abs(min(afb_results.delta_C9_values))
   
    afb_ax = axs_2x1.flat[0]
    s5_ax = axs_2x1.flat[1]

    cmap = plt.cm.coolwarm
    norm = mpl.colors.CenteredNorm(
        vcenter=0, 
        halfrange=get_colorbar_halfrange(afb_results=afb_results, s5_results=s5_results),
    )

    plot_afb(afb_results=afb_results, ax=afb_ax, cmap=cmap, norm=norm, alpha=alpha)
    plot_s5(s5_results=s5_results, ax=s5_ax, cmap=cmap, norm=norm, alpha=alpha)

    s5_ax.get_legend().remove()
    afb_ax.get_xlabel().remove()

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axs_2x1, 
        orientation='vertical', 
        label=r'$\delta C_9$',
    )


    
def plot_efficiency(
    efficiency_result,
    ax,
    ylimits=(0, 0.5),
):

    ax.scatter(
        efficiency_result.bin_middles, 
        efficiency_result.efficiencies,
        color="purple",
        edgecolors='none',
        s=10,
    )
    ax.errorbar(
        efficiency_result.bin_middles,
        efficiency_result.efficiencies,
        yerr=efficiency_result.errors,
        fmt="none",
        capsize=0,
        elinewidth=0.5, 
        ecolor="black"
    )
    
    ax.set_ylim(ylimits)


def plot_resolution(
    resolution,
    name_of_variable,
    xlimits,
    num_bins
):
    
    def save_plot():
        pass
        
    fig, ax = plt.subplots()

    ax.hist(
        resolution,
        label=stats_legend(resolution),
        bins=num_bins,
        range=xlimits,
        color="purple",
        histtype="step",
    )

    ax.legend()
    ax.set_xlim(xlimits)

    xlabels = {
        Names_of_Variables().q_squared : r"$q^2 - q^{2\;MC}$",
        Names_of_Variables().costheta_mu : r"$\cos\theta_\mu - \cos\theta_\mu^{MC}$",
        Names_of_Variables().cos_k : r"$\cos\theta_K - \cos\theta_K^{MC}$",
        Names_of_Variables().chi : r"$\chi - \chi^{MC}$"
    }
    ax.set_xlabel(xlabels[name_of_variable])

    save_plot()


def plot_image_slices(
    image,
    save_path,
    num_slices=3, 
    cmap=plt.cm.magma,
    note=None,
):  
    
    """
    Plot slices of a B->K*ll dataset image.

    Slices are along the chi-axis (axis 2)
    and might not be evenly spaced.
    """

    def xy_plane_at(z_position):

        x, y = numpy.indices(
            (
                axis_dimension_from_cartesian["x"] + 1, 
                axis_dimension_from_cartesian["y"] + 1
            )
        )
        z = numpy.full(
            (
                axis_dimension_from_cartesian["x"] + 1, 
                axis_dimension_from_cartesian["y"] + 1
            ), 
            z_position
        )
        return x, y, z
    
    def plot_slice(z_index):

        x, y, z = xy_plane_at(z_index) 
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False
        )

    def plot_outline(z_index, offset=0.3):

        x, y, z = xy_plane_at(z_index - offset)
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, 
            cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2"
        )

    fig = plt.figure()
    ax_3d = fig.add_subplot(projection="3d")

    image = image.squeeze()
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    colors = cmap(norm(image))
    
    axis_index_from_cartesian = {
        "x": 0,
        "y": 1,
        "z": 2
    }
    axis_dimension_from_cartesian = {
        "x": image.shape[axis_index_from_cartesian["x"]],
        "y": image.shape[axis_index_from_cartesian["y"]],
        "z": image.shape[axis_index_from_cartesian["z"]]
    }

    z_indices = numpy.linspace( 
        start=0, 
        stop=axis_dimension_from_cartesian["z"]-1, 
        num=num_slices, 
        dtype=int  # forces integer indices
    ) 

    for i in z_indices:
        plot_outline(i)
        plot_slice(i)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax_3d, 
        location="left", 
        shrink=0.5, 
        pad=-0.05
    )

    cbar.set_label(r"${q^2}$ (Avg.)", size=11)
    ax_labels = {
        "x": r"$\cos\theta_\mu$",
        "y": r"$\cos\theta_K$",
        "z": r"$\chi$"
    }
    ax_3d.set_xlabel(ax_labels["x"], labelpad=0)
    ax_3d.set_ylabel(ax_labels["y"], labelpad=0)
    ax_3d.set_zlabel(ax_labels["z"], labelpad=-3)
    ax_3d.tick_params(pad=0.3)
    ax_3d.set_box_aspect(None, zoom=0.85)
    if note:
        ax_3d.set_title(f"{note}", loc="left", y=0.85)

    plt.savefig(save_path, bbox_inches="tight")    
    plt.close()


def plot_linearity(
    lin_results,
    xlim=(-2.25, 1.35), 
    ylim=(-2.25, 1.35),
    ax=None,
):
    
    def plot_diagonal_reference_line(unique_labels_numpy_array):

        buffer = 0.05
        
        ticks = numpy.linspace(
            start=(numpy.min(unique_labels_numpy_array) - buffer),
            stop=(numpy.max(unique_labels_numpy_array) + buffer),
            num=2 
        )
        
        ax.plot(
            ticks, 
            ticks,
            label=None,
            color="grey",
            linewidth=0.5,
            zorder=0,
        )

    if ax is None:
        _, ax = plt.subplots()

    unique_labels, avgs, stds = lin_results

    ax.scatter(
        unique_labels, 
        avgs, 
        label="Avg.", 
        color="firebrick", 
        s=6, 
        zorder=5
    )
    
    ax.errorbar(
        unique_labels, 
        avgs, 
        yerr=stds, 
        fmt="none", 
        elinewidth=0.5, 
        capsize=0.5, 
        color="black", 
        label="Stdev.", 
        zorder=10
    )

    plot_diagonal_reference_line(unique_labels)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks(ticks=[-2, 0, 1.1])
    ax.set_yticks(ticks=[-2, 0, 1.1])


    

def plot_loss_curves(
    model_settings,
    start_epoch=0, 
    log_scale=False,
    note=None,
):
    
    loss_table = Loss_Table()
    loss_table.load(model_settings.paths.make_path_to_loss_table_file())

    epochs_to_plot = loss_table.epochs[start_epoch:]
    train_losses_to_plot = loss_table.train_losses[start_epoch:]
    eval_losses_to_plot = loss_table.eval_losses[start_epoch:]

    _, ax = plt.subplots()

    ax.plot(
        epochs_to_plot, 
        train_losses_to_plot, 
        label="Training Loss"
    )
    ax.plot(
        epochs_to_plot, 
        eval_losses_to_plot, 
        label="Eval. Loss"
    )

    if log_scale: ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("Epoch")
    note = (
        (
            f"Loss curves: {model_settings.name}, {model_settings.training.training_dataset_settings.common.level}, ", 
            f"events/set: {model_settings.training.training_dataset_settings.set.num_signal_events_per_set}"
        )
        if model_settings.name in Names_of_Models().set_based
        else f"Loss curves: {model_settings.name}, {model_settings.training.training_dataset_settings.common.level}"
        if model_settings.name in Names_of_Models().event_based
        else None
    )
    if note is None: raise ValueError
    add_plot_note(ax=ax, text=note)
    
    plt.savefig(model_settings.paths.make_path_to_loss_curves_plot(), bbox_inches="tight")
    plt.close()



def plot_log_probability_distribution_examples(
    log_probabilities, 
    binned_labels, 
    bin_map,
    ax
):

    unbinned_labels = bin_map[binned_labels]
    example_log_probabilities_standard_model = log_probabilities[unbinned_labels==delta_C9_value_standard_model][0].cpu()
    example_log_probabilities_new_physics = log_probabilities[unbinned_labels==delta_C9_value_new_physics][0].cpu()

    ax.axvline(x=delta_C9_value_standard_model, color='black', label=None, ls='--')
    ax.axvline(x=delta_C9_value_new_physics, color='black', label=None, ls=':')

    ax.scatter(bin_map.cpu(), example_log_probabilities_standard_model, label="SM", color="blue", s=2.5, marker="s")
    ax.scatter(bin_map.cpu(), example_log_probabilities_new_physics, label="NP", color="red", s=3, marker="o")

    ax.set_xticks([-2, -0.82, 0, 1.1])

    

def plot_sensitivity(
    sensitivity_test_results,
    model_settings,
    dataset_settings,
    path_to_plots_dir,
    bins=50, 
    xbounds=(-1.5, 0), 
    ybounds=(0, 200), 
):
    
    def add_label_marker(ax, label, ybounds):
        ax.vlines(
            x=label,
            ymin=ybounds[0],
            ymax=ybounds[1],
            color="red",
            label=r"Actual $\delta C_9$ " + f"({label})"
        )

    def add_prediction_average_marker(ax, average, ybounds):
        ax.vlines(
            x=average,
            ymin=ybounds[0],
            ymax=ybounds[1],
            color="red",
            linestyles="--",
            label=f"Average: {average.round(decimals=3)}"
        )

    def add_prediction_standard_deviation_marker(ax, average, standard_deviation):
        ax.hlines(
            y=(ybounds[1] / 10),
            xmin=average,
            xmax=average+standard_deviation,
            color="orange",
            linestyles="dashdot",
            label=(f"Standard deviation: {standard_deviation.round(decimals=3)}")
        )

    _, ax = plt.subplots()

    ax.hist(sensitivity_test_results.predicted_values, bins=bins, range=xbounds)

    add_label_marker(ax=ax, label=sensitivity_test_results.label, ybounds=ybounds)
    add_prediction_average_marker(ax=ax, average=sensitivity_test_results.avg, ybounds=ybounds)
    add_prediction_standard_deviation_marker(
        ax=ax, 
        average=sensitivity_test_results.avg, 
        standard_deviation=sensitivity_test_results.std
    )
    ax.set_xlabel(r"Predicted $\delta C_9$")
    ax.set_xbound(*xbounds)
    ax.set_ybound(*ybounds)
    ax.legend()
    note = (
        f"Model: {model_settings.name}, Level: {dataset_settings.common.level}", 
        f"\nNumber of bootstraps: {dataset_settings.set.num_sets_per_label}, Events per bootstrap: {dataset_settings.set.num_events_per_set}"
    )
    add_plot_note(ax=ax, text=note)
    
    save_model_evaluation_plot(
        type="sens",
        model_settings=model_settings,
        dataset_settings=dataset_settings,
        path_to_plots_dir=path_to_plots_dir
    )
    plt.close()



def plot_distribution(data, ax, delta_C9_value, cmap, norm, xlimits, num_bins):

    is_standard_model = (delta_C9_value == 0)

    color = cmap(norm(delta_C9_value)) if not is_standard_model else "dimgrey"
    linestyle = 'solid' if not is_standard_model else (0, (1, 1))
    label = None if not is_standard_model else r"SM ($\delta C_9 = 0$)"
    zorder = None if not is_standard_model else 10
    
    ax.hist(
        data, 
        range=xlimits,
        histtype="step",
        density=True, 
        bins=num_bins, 
        color=color,
        linestyle=linestyle,
        label=label,
        zorder=zorder
    )


def plot_variable(dataframe, name_of_variable, ax, cmap, norm, num_bins):

    assert name_of_variable in Names_of_Variables().tuple_
    
    xlabels = {
        Names_of_Variables().q_squared : r"$q^2$ [GeV$^2$]",
        Names_of_Variables().cos_theta_mu : r"$\cos\theta_\mu$",
        Names_of_Variables().cos_k : r"$\cos\theta_K$",
        Names_of_Variables().chi : r"$\chi$"
    }
    xlimits = {
        Names_of_Variables().q_squared : (0, 20),
        Names_of_Variables().cos_theta_mu : (-1, 1),
        Names_of_Variables().cos_k : (-1, 1),
        Names_of_Variables().chi : (0, 2*pi),
    }
    ylims = {
        Names_of_Variables().q_squared : (0, 0.14),
        Names_of_Variables().cos_theta_mu : (0, 0.75),
        Names_of_Variables().cos_k : (0, 0.85),
        Names_of_Variables().chi : (0, 0.195),
    }

    for delta_C9_value, dataframe_subset in dataframe.groupby(Names_of_Labels().unbinned):
        plot_distribution(
            data=dataframe_subset[name_of_variable],
            ax=ax,
            delta_C9_value=delta_C9_value,
            cmap=cmap,
            norm=norm,
            xlimits=xlimits[name_of_variable],
            num_bins=num_bins
        )

    if name_of_variable == Names_of_Variables().chi:
        ax.set_xticks(
            [0, pi, 2*pi], 
            ["0", r"$\pi$", r"$2 \pi$"],
        )
    ax.set_xlabel(xlabels[name_of_variable])
    ax.set_ylim(ylims[name_of_variable])
    ax.locator_params(axis='y', tight=True, nbins=2)


def plot_all_variables(fig, axs_2x2, dataframe, num_bins):

    cmap = plt.cm.coolwarm
    norm = mpl.colors.CenteredNorm(
        vcenter=0, 
        halfrange=abs(dataframe[Names_of_Labels().unbinned].min())
    )

    for variable, ax in zip(Names_of_Variables().tuple_, axs_2x2.flat):
        plot_variable(
            dataframe=dataframe,
            name_of_variable=variable,
            ax=ax,
            cmap=cmap,
            norm=norm,
            num_bins=num_bins
        )

    axs_2x2[0,0].legend()
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=axs_2x2, 
        orientation='vertical', 
    )
    cbar.set_label(r'$\delta C_9$', size=11)
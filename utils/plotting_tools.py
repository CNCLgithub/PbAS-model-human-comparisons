import numpy as np

from scipy.stats import spearmanr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats



class PlottingTools():
    def __init__(self):
        pass

    def multiple_barplots(self, data, x_labels, panel_row_count, panel_column_count,
                          filename, fig_height, fig_width, colors=None, hatches=None, ylimits=None, verbose=False, pdf=True, summary_data=True, set_title=[]):

        all_count = panel_row_count * panel_column_count

        if colors is not None:
            if len(colors.shape) < 3:
                colors = np.expand_dims(colors, 0)
                colors = np.repeat(colors, all_count, axis=0)
                for idx in range(all_count-1):
                    colors[idx+1] = colors[0]

        fig, axs = plt.subplots(panel_row_count, panel_column_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        for idx in range(all_count):

            if summary_data == True:
                #Outer conditions - inner conditions - mean sem
                means = data[:, idx, 0]
                high_ci = means + data[:, idx, 1]
                low_ci = means - data[:, idx, 1]
                show_error_bars = True
            else:
                means = np.array([np.mean(x) for x in data[:, idx, :]])
                if data.shape[2] == 1: # if it is just mean values
                    show_error_bars = False
                else:
                    high_ci = np.array([np.percentile(x, 97.5) for x in data[:, idx, :]])
                    low_ci = np.array([np.percentile(x, 2.5) for x in data[:, idx, :]])
                    show_error_bars = True
                    
            if verbose == True:
                print(means)

            if all_count == 1:
                ax = axs
            else:
                ax = axs[idx]
            xs = np.arange(means.shape[0])

            if show_error_bars:
                if colors is None:
                    ax.bar(xs, means,
                           yerr=np.vstack([high_ci - means, means - low_ci]))
                elif hatches is None:
                    ax.bar(xs, means,
                           yerr=np.vstack([high_ci - means, means - low_ci]),
                           color=colors[idx])
                else:
                    bar_curr = ax.bar(xs, means,
                           yerr=np.vstack([high_ci - means, means - low_ci]),
                                      color=colors[idx])
                    for b, bar in enumerate(bar_curr):
                        bar.set_hatch(hatches[b])

            else:
                if hatches is None:
                    ax.bar(xs, means,
                           color=colors[idx])
                else:
                    bar_curr = ax.bar(xs, means,
                                      color=colors[idx])
                    for b, bar in enumerate(bar_curr):
                        bar.set_hatch(hatches[b])
                    


            ax.tick_params(labelsize=12)
            if x_labels is not None:
                ax.set_xticks(xs)
                ax.set_xticklabels(x_labels, rotation=40)
            else:
                ax.set_xticks([])
            if ylimits is not None:
                ax.set_ylim(ylimits)
            else:
                ax.set_ylim((0, 1))
            if idx > 0:
                ax.set_yticks([])
            if idx == 0 and ylimits is None:
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            if set_title != []:
                ax.set_title(set_title[idx])

        all_axes = fig.get_axes()

        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if pdf == True:
            plot_path = './output/' + filename + '.pdf'
        else:
            plot_path = './output/' + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


    def multiple_violins(self, data, x_labels, panel_row_count, panel_column_count,
                         filename, fig_height, fig_width, colors=None, ylimits=None, verbose=False, pdf=True, summary_data=True, set_title=[]):

        fig, axs = plt.subplots(panel_row_count, panel_column_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        all_count = panel_row_count * panel_column_count
        for idx in range(all_count):
            medians = np.array([np.median(x) for x in data[:, idx, :]])
            quartile_top = np.array([np.percentile(x, 75) for x in data[:, idx, :]])
            quartile_bottom = np.array([np.percentile(x, 25) for x in data[:, idx, :]])
                    
            if verbose == True:
                print(medians)

            if all_count == 1:
                ax = axs
            else:
                ax = axs[idx]
            xs = np.arange(1, medians.shape[0]+1)
            if colors is None:
                ax.violinplot(list(np.squeeze(data[:, idx,:])))
            else:
                parts = ax.violinplot(list(np.squeeze(data[:, idx, :])), showextrema=False, showmedians=False)
                for pc_index, pc in enumerate(parts['bodies']):
                    pc.set_color(colors[pc_index])
                    pc.set_alpha(1)
                ax.scatter(xs, medians, marker='o', color='gray', s=30, zorder=3)
                ax.vlines(xs, quartile_bottom, quartile_top, color='k', linestyle='-', lw=4)

            ax.tick_params(labelsize=12)
            if x_labels is not None:
                ax.set_xticks(xs)
                ax.set_xticklabels(x_labels, rotation=40)
            else:
                ax.set_xticks([])
            if ylimits is not None:
                ax.set_ylim(ylimits)
            else:
                ax.set_ylim((0, 1))
            if idx > 0:
                ax.set_yticks([])
            if idx == 0 and ylimits is None:
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            if set_title != []:
                ax.set_title(set_title[idx])

        all_axes = fig.get_axes()

        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if pdf == True:
            plot_path = './output/' + filename + '.pdf'
        else:
            plot_path = './output/' + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


    def multiple_lines(self, means, sems, filename, fig_height, fig_width, horizontal_lines=None, labels=None, y_limits=None, verbose=False, colors=None, linetypes=None, markers=None, stepsize=None, start=None, end=None, shade=None, set_markers=None, marker_colors=None, pdf=True):
        means = np.array(means)
        sems = np.array(sems)
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        if verbose == True:
            print(means)
        
        ax = axs
        xs = np.arange(means.shape[1])


        for s in range(means.shape[0]):
            if shade is not None:
                if shade[s] == True:
                    ax.plot(xs, means[s], color=colors[s], linestyle=linetypes[s], marker=markers[s])
                    ax.fill_between(xs, means[s] - sems[s], means[s] + sems[s],
                                    alpha=0.25, edgecolor='white', facecolor=colors[s])
                else:
                    ax.errorbar(xs, means[s],
                                yerr=np.vstack([sems[s], sems[s]]),
                                color=colors[s], linestyle=linetypes[s],
                                marker=markers[s])
            else:
                ax.plot(xs, means[s], color=colors[s])

            if horizontal_lines is not None:
                #ax.axhline(horizontal_lines[s], color=colors[s], linestyle='dashed')
                ax.hlines(horizontal_lines[s], start, end, color="black", linestyle=(1, (5, 10)))

            if set_markers is not None:
                ax.scatter(set_markers[s], 0.01, color=marker_colors[s], marker='v', s=100)

            ax.set_ylim((0, 0.6))
        """
        for s in range(means.shape[0]):
            if colors is not None:
                ax.errorbar(xs, means[s],
                            yerr=np.vstack([sems[s], sems[s]]),
                            color=colors[s], linestyle=linetypes[s],
                            marker=markers[s])
            else:
                ax.errorbar(xs, means[s],
                            yerr=np.vstack([sems[s], sems[s]]))
        """
        ax.tick_params(labelsize=12)
        ax.set_xticks(range(means.shape[1]))
        if labels is not None:
            ax.set_xticklabels(labels)

        if y_limits is not None:
            ax.set_ylim(y_limits)

        #start, end = ax.get_xlim()
        if stepsize is not None:
            ticks = np.arange(start, end, stepsize)
            ax.xaxis.set_ticks(ticks)
            ticks[0] += 1
            ax.set_xticklabels(ticks, rotation=0)

        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if pdf == True:
            plot_path = './output/' + filename + '.pdf'
        else:
            plot_path = './output/' + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)


    def multiple_lines_different_colors(self, means, sems, filename, fig_height, fig_width, horizontal_lines=None, labels=None, y_limits=None, verbose=False, colors=None, linetypes=None, markers=None, stepsize=None, start=None, end=None, shade=None, set_markers=None, marker_colors=None, pdf=False):
        means = np.array(means)
        sems = np.array(sems)
        fig, axs = plt.subplots(1, 1)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        if verbose == True:
            print(means)
        
        ax = axs
        xs = np.arange(means.shape[1])

        for s in range(means.shape[0]):
            if shade is not None:
                if shade[s] == True:
                    ax.plot(xs, means[s], color=colors[s], linestyle=linetypes[s], marker=markers[s])
                    ax.fill_between(xs, means[s] - sems[s], means[s] + sems[s],
                                    alpha=0.25, edgecolor='white', facecolor=colors[s])
                else:
                    ax.errorbar(xs, means[s],
                                yerr=np.vstack([sems[s], sems[s]]),
                                color=colors[s], linestyle=linetypes[s],
                                marker=markers[s])
            else:
                for j in range(means.shape[2]):
                    ax.plot(xs, means[s, :, j], color=colors[j], linewidth=2)

            if horizontal_lines is not None:
                #ax.axhline(horizontal_lines[s], color=colors[s], linestyle='dashed')
                ax.hlines(horizontal_lines[s], start, end, color="black", linestyle=(1, (5, 5)), linewidth=5)

            if set_markers is not None:
                ax.scatter(set_markers[s], 0.01, color=marker_colors[s], marker='v', s=100)

            ax.set_ylim((0, 0.6))

        ax.tick_params(labelsize=20)
        ax.set_xticks(range(means.shape[1]))
        if labels is not None:
            ax.set_xticklabels(labels)

        if y_limits is not None:
            ax.set_ylim(y_limits)

        #start, end = ax.get_xlim()
        if stepsize is not None:
            ticks = np.arange(start, end, stepsize)
            ax.xaxis.set_ticks(ticks)
            ticks[0] += 1
            ax.set_xticklabels(ticks, rotation=0)

        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if pdf == True:
            plot_path = './output/' + filename + '.pdf'
        else:
            plot_path = './output/' + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)




    def scatter_plot_multipane(self, data_1, data_2, filename, fig_height, fig_width, mask=None, colors=None, verbose=False, pdf=True, submask=None, xlimits=None, ylimits=None, regline=None, dotsize=10):

        panel_column_count = np.max(mask) + 1

        panel_row_count = 1
        fig, axs = plt.subplots(panel_row_count, panel_column_count)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)

        panel_counter = 0
        for i in range(panel_row_count):
            for j in range(panel_column_count):
                if panel_row_count == panel_column_count == 1:
                    ax = axs
                else:
                    ax = axs[j]
                
                    
                if regline is not None:
                    ### Source: https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot
                    x = data_1[mask == j]
                    y = data_2[mask == j]
                    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
                    y_model = np.polyval(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients
                        
                    # Statistics
                    n = y.size                                           # number of observations
                    m = p.size                                                 # number of parameters
                    dof = n - m                                                # degrees of freedom
                    t = stats.t.ppf(0.975, n - m)                              # t-statistic; used for CI and PI bands
                    # Estimates of Error in Data/Model
                    resid = y - y_model                                        # residuals; diff. actual data from predicted values
                    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
                    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
                    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

                    x2 = np.linspace(np.min(x), np.max(x), 100)
                    y2 = np.polyval(p, x2)
                    ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5) 

                    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                    ax.fill_between(x2, y2 + ci, y2 - ci, color='gray', edgecolor="", alpha=0.5)

                if submask is not None:
                    a = data_1[mask == j]
                    b = data_2[mask == j]
                    for k in range(np.max(submask) + 1):
                        ax.scatter(a[submask == k], b[submask == k], color=colors[k], s=dotsize)
                        if verbose == True:
                            print(np.corrcoef(a[submask==k], b[submask==k])[0,1])
                            #print(spearmanr(a[submask==k], b[submask==k]))

                else:
                    if verbose == True:
                        print(np.corrcoef(data_1[mask==j], data_2[mask==j])[0,1])
                        #print(spearmanr(data_1[mask==j], data_2[mask==j]))
                    ax.scatter(data_1[mask == j], data_2[mask == j], color=colors[j], s=dotsize)

                if ylimits is not None:
                    ax.set_ylim(ylimits)
                if xlimits is not None:
                    ax.set_xlim(xlimits)


        all_axes = fig.get_axes()
        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ax.is_last_row():
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)

        if pdf == True:
            plot_path = './output/' + filename + '.pdf'
        else:
            plot_path = './output/' + filename + '.png'
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

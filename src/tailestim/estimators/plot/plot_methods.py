import os
import numpy as np
from matplotlib import pyplot as plt

def make_plots(ordered_data, results, output_file_path, alpha=0.6,
               bootstrap_flag=True, diagn_plots=False, theta1=0.01, theta2=0.99,
               verbose=False, noise_flag=True, savedata=False):
    """
    Create plots for tail index estimation.
    
    Parameters
    ----------
    ordered_data : numpy.ndarray
        Array for which tail index estimation is performed (decreasing ordering required).
    results : dict
        Dictionary containing the results from fit_estimators.
    output_file_path : str or None, optional
        File path to which plots should be saved. If None, the figure is not saved.
    alpha : float, optional
        Parameter controlling the amount of "smoothing" for the kernel-type estimator. 
        Should be greater than 0.5, default is 0.6.
    bootstrap_flag : bool, optional
        Flag to switch on/off double-bootstrap procedure, default is True.
    diagn_plots : bool, optional
        Flag to switch on/off generation of AMSE diagnostic plots, default is False.
    theta1 : float, optional
        Lower bound of plotting range, defined as k_min = ceil(n^theta1).
        Overwritten if plots behave badly within the range, default is 0.01.
    theta2 : float, optional
        Upper bound of plotting range, defined as k_max = floor(n^theta2).
        Overwritten if plots behave badly within the range, default is 0.99.
    verbose : bool, optional
        Flag controlling bootstrap verbosity, default is False.
    noise_flag : bool, optional
        Switch on/off uniform noise in range [-5*10^(-p), 5*10^(-p)] that is added to each
        data point. Used for integer-valued sequences with p = 1, default is True.
    savedata : bool, optional
        Flag to save data files in the directory with plots, default is False.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray or None
    """

    # Extract results for plotting
    x_pdf, y_pdf = results['pdf']['x'], results['pdf']['y']
    x_ccdf, y_ccdf = results['ccdf']['x'], results['ccdf']['y']
    ordered_data = results['ordered_data']
    if noise_flag:
        discrete_ordered_data = results['discrete_ordered_data']
    
    # Pickands results
    k_p_arr, xi_p_arr = results['pickands']['k_arr_'], results['pickands']['xi_arr_']
    
    # Smooth Hill results
    k_sh_arr, xi_sh_arr = results['smooth_hill']['k_arr_'], results['smooth_hill']['xi_arr_']
    
    # Hill results
    k_h_arr, xi_h_arr = results['hill']['k_arr_'], results['hill']['xi_arr_']
    if bootstrap_flag and 'k_star_' in results['hill']:
        k_h_star = results['hill']['k_star_']
        xi_h_star = results['hill']['xi_star_']
        x1_h_arr = results['hill']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_h_amse = results['hill']['bootstrap_results_']['first_bootstrap_']['amse_']
        k1_h = results['hill']['bootstrap_results_']['first_bootstrap_']['k_min_']
        max_h_index1 = results['hill']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_h_arr = results['hill']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_h_amse = results['hill']['bootstrap_results_']['second_bootstrap_']['amse_']
        k2_h = results['hill']['bootstrap_results_']['second_bootstrap_']['k_min_']
        max_h_index2 = results['hill']['bootstrap_results_']['second_bootstrap_']['max_index_']
    else:
        k_h_star = None
        xi_h_star = None
        x1_h_arr, n1_h_amse, k1_h, max_h_index1 = None, None, None, None
        x2_h_arr, n2_h_amse, k2_h, max_h_index2 = None, None, None, None
    
    # Moments results
    k_m_arr, xi_m_arr = results['moments']['k_arr_'], results['moments']['xi_arr_']
    if bootstrap_flag and 'k_star_' in results['moments']:
        k_m_star = results['moments']['k_star_']
        xi_m_star = results['moments']['xi_star_']
        x1_m_arr = results['moments']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_m_amse = results['moments']['bootstrap_results_']['first_bootstrap_']['amse_']
        k1_m = results['moments']['bootstrap_results_']['first_bootstrap_']['k_min_']
        max_m_index1 = results['moments']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_m_arr = results['moments']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_m_amse = results['moments']['bootstrap_results_']['second_bootstrap_']['amse_']
        k2_m = results['moments']['bootstrap_results_']['second_bootstrap_']['k_min_']
        max_m_index2 = results['moments']['bootstrap_results_']['second_bootstrap_']['max_index_']
    else:
        k_m_star = None
        xi_m_star = None
        x1_m_arr, n1_m_amse, k1_m, max_m_index1 = None, None, None, None
        x2_m_arr, n2_m_amse, k2_m, max_m_index2 = None, None, None, None
    
    # Kernel results
    k_k_arr, xi_k_arr = results['kernel']['k_arr_'], results['kernel']['xi_arr_']
    if bootstrap_flag and 'k_star_' in results['kernel']:
        k_k_star = results['kernel']['k_star_']
        xi_k_star = results['kernel']['xi_star_']
        x1_k_arr = results['kernel']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_k_amse = results['kernel']['bootstrap_results_']['first_bootstrap_']['amse_']
        h1 = results['kernel']['bootstrap_results_']['first_bootstrap_']['h_min']
        max_k_index1 = results['kernel']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_k_arr = results['kernel']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_k_amse = results['kernel']['bootstrap_results_']['second_bootstrap_']['amse_']
        h2 = results['kernel']['bootstrap_results_']['second_bootstrap_']['h_min']
        max_k_index2 = results['kernel']['bootstrap_results_']['second_bootstrap_']['max_index_']
        k_k1_star = results['kernel'].get('k_k1_star')
    else:
        k_k_star = None
        xi_k_star = None
        x1_k_arr, n1_k_amse, h1, max_k_index1 = None, None, None, None
        x2_k_arr, n2_k_amse, h2, max_k_index2 = None, None, None, None
        k_k1_star = None
    # Setup for saving data if needed
    if savedata:
        output_dir = os.path.dirname(os.path.realpath(output_file_path))
        output_name = os.path.splitext(os.path.basename(output_file_path))[0]
        
        # Save PDF data
        with open(os.path.join(output_dir+"/"+output_name+"_pdf.dat"), "w") as f:
            for i in range(len(x_pdf)):
                f.write(str(x_pdf[i]) + " " + str(y_pdf[i]) + "\n")
                
        # Save CCDF data
        with open(os.path.join(output_dir+"/"+output_name+"_ccdf.dat"), "w") as f:
            for i in range(len(x_ccdf)):
                f.write(str(x_ccdf[i]) + " " + str(y_ccdf[i]) + "\n")
    
    # Save additional data files if needed
    if savedata:
        # Save Pickands data
        with open(os.path.join(output_dir+"/"+output_name+"_pickands.dat"), "w") as f:
            for i in range(len(k_p_arr)):
                f.write(str(k_p_arr[i]) + " " + str(xi_p_arr[i]) + "\n")
        
        # Save smooth Hill data
        with open(os.path.join(output_dir+"/"+output_name+"_sm_hill.dat"), "w") as f:
            for i in range(len(k_sh_arr)):
                f.write(str(k_sh_arr[i]) + " " + str(xi_sh_arr[i]) + "\n")
        
        # Save Hill data
        with open(os.path.join(output_dir+"/"+output_name+"_adj_hill_plot.dat"), "w") as f:
            for i in range(len(k_h_arr)):
                f.write(str(k_h_arr[i]) + " " + str(xi_h_arr[i]) + "\n")
        if bootstrap_flag and k_h_star is not None:
            with open(os.path.join(output_dir+"/"+output_name+"_adj_hill_estimate.dat"), "w") as f:
                f.write(str(k_h_star) + " " + str(xi_h_star) + "\n")
        
        # Save moments data
        with open(os.path.join(output_dir+"/"+output_name+"_mom_plot.dat"), "w") as f:
            for i in range(len(k_m_arr)):
                f.write(str(k_m_arr[i]) + " " + str(xi_m_arr[i]) + "\n")
        if bootstrap_flag and k_m_star is not None:
            with open(os.path.join(output_dir+"/"+output_name+"_mom_estimate.dat"), "w") as f:
                f.write(str(k_m_star) + " " + str(xi_m_star) + "\n")
        
        # Save kernel data
        with open(os.path.join(output_dir+"/"+output_name+"_kern_plot.dat"), "w") as f:
            for i in range(len(k_k_arr)):
                f.write(str(k_k_arr[i]) + " " + str(xi_k_arr[i]) + "\n")
        if bootstrap_flag and k_k1_star is not None:
            with open(os.path.join(output_dir+"/"+output_name+"_kern_estimate.dat"), "w") as f:
                f.write(str(k_k_arr[k_k1_star]) + " " + str(xi_k_arr[k_k1_star]) + "\n")
    
    # plotting part
    if verbose:
        print("Making plots...")

    fig, axes = plt.subplots(3, 2, figsize = (12, 16))
    for ax in axes.reshape(-1):
        ax.tick_params(direction='out', length=6, width=1.5,
                       labelsize = 12, which = 'major')
        ax.tick_params(direction='out', length=3, width=1, which = 'minor')
        [i.set_linewidth(1.5) for i in ax.spines.values()]

    # plot PDF
    axes[0,0].set_xlabel(r"Degree $k$", fontsize = 20)
    axes[0,0].set_ylabel(r"$P(k)$", fontsize = 20)
    axes[0,0].loglog(x_pdf, y_pdf, color = "#386cb0", marker = "s",
                     lw = 1.5, markeredgecolor = "black")

    # plot CCDF
    axes[0,1].set_xlabel(r"Degree $k$", fontsize = 20)
    axes[0,1].set_ylabel(r"$\bar{F}(k)$", fontsize = 20)
    axes[0,1].set_xscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].step(x_ccdf, y_ccdf, color = "#386cb0", lw = 1.5)
    
    # draw scalings
    # Draw Hill scaling only if bootstrap is enabled and k_h_star is available
    if bootstrap_flag and k_h_star is not None:
        if noise_flag:
            xmin = discrete_ordered_data[k_h_star]
        else:
            xmin = ordered_data[k_h_star]
        x = x_ccdf[np.where(x_ccdf >= xmin)]
        if len(x) > 0:  # Make sure we have points to plot
            l = np.mean(y_ccdf[np.where(x == xmin)])
            alpha = 1./xi_h_star
            if xi_h_star > 0:
                axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#fb8072', ls = '--', lw = 2,
                       label = r"Adj. Hill Scaling $(\alpha="+\
                       str(np.round(1./xi_h_star, decimals = 3))+r")$")
                axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                               color = "#fb8072", ls = 'none', marker = 'o',
                               markerfacecolor = 'none', markeredgecolor = "#fb8072",
                               markeredgewidth = 3, markersize = 10)
    # Draw Moments scaling only if bootstrap is enabled and k_m_star is available
    if bootstrap_flag and k_m_star is not None:
        if noise_flag:
            xmin = discrete_ordered_data[k_m_star]
        else:
            xmin = ordered_data[k_m_star]
        x = x_ccdf[np.where(x_ccdf >= xmin)]
        if len(x) > 0:  # Make sure we have points to plot
            l = np.mean(y_ccdf[np.where(x == xmin)])
            alpha = 1./xi_m_star
            if xi_m_star > 0:
                axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#8dd3c7', ls = '--', lw = 2,
                       label = r"Moments Scaling $(\alpha="+\
                       str(np.round(1./xi_m_star, decimals = 3))+r")$")
                axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                               color = "#8dd3c7", ls = 'none', marker = 'o',
                               markerfacecolor = 'none', markeredgecolor = "#8dd3c7",
                               markeredgewidth = 3, markersize = 10)
    # Draw Kernel scaling only if bootstrap is enabled and k_k_star is available
    if bootstrap_flag and k_k_star is not None:
        if noise_flag:
            xmin = discrete_ordered_data[k_k_star]
        else:
            xmin = ordered_data[k_k_star]
        
        x = x_ccdf[np.where(x_ccdf >= xmin)]
        if len(x) > 0:  # Make sure we have points to plot
            l = np.mean(y_ccdf[np.where(x == xmin)])
            alpha = 1./xi_k_star
            if xi_k_star > 0:
                axes[0,1].plot(x, [l*(float(xmin)/k)**alpha for k in x],
                       color = '#fdb462', ls = '--', lw = 2,
                       label = r"Kernel Scaling $(\alpha="+\
                       str(np.round(1./xi_k_star, decimals = 3))+r")$")
                axes[0,1].plot((x[-1]), [l*(float(xmin)/x[-1])**(alpha)],
                               color = "#fdb462", ls = 'none', marker = 'o',
                               markerfacecolor = 'none', markeredgecolor = "#fdb462",
                               markeredgewidth = 3, markersize = 10)
    axes[0,1].legend(loc = 'best')

    # define min and max order statistics to plot
    min_k = int(np.ceil(len(k_h_arr)**theta1)) - 1
    max_k = int(np.floor(len(k_h_arr)**theta2)) - 1
    # check if estimators' values are not too off in these bounds
    min_k_index = (np.abs(k_sh_arr - min_k)).argmin()
    max_k_index = (np.abs(k_sh_arr - max_k)).argmin()
    if (xi_sh_arr[min_k_index] <= -3 or xi_sh_arr[min_k_index] >= 3):
        indices_to_plot_sh = np.where((xi_sh_arr <= 3) & (xi_sh_arr >= -3))
    elif (xi_sh_arr[max_k_index] <= -3 or xi_sh_arr[max_k_index] >= 3):
        indices_to_plot_sh = np.where((xi_sh_arr <= 3) & (xi_sh_arr >= -3))
    else:
        indices_to_plot_sh = np.where((k_sh_arr <= max_k) & (k_sh_arr >= min_k))
    axes[1,0].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[1,0].set_ylabel(r"Estimated $\hat{\xi}$", fontsize = 20)    
    # plot smooth Hill
    
    axes[1,0].plot(k_sh_arr[indices_to_plot_sh], xi_sh_arr[indices_to_plot_sh],
                   color = "#b3de69", alpha = 0.8, label = "Smooth Hill",
                   zorder = 10)

    # plot adjusted Hill
    # check if estimators' values are not too off in these bounds
    if (xi_h_arr[min_k-1] <= -3 or xi_h_arr[min_k-1] >= 3):
        indices_to_plot_h = np.where((xi_h_arr <= 3) & (xi_h_arr >= -3))
    elif (xi_h_arr[max_k-1] <= -3 or xi_h_arr[max_k-1] >= 3):
        indices_to_plot_h = np.where((xi_h_arr <= 3) & (xi_h_arr >= -3))
    else:
        indices_to_plot_h = np.where((k_h_arr <= max_k) & (k_h_arr >= min_k))
    axes[1,0].plot(k_h_arr[indices_to_plot_h], xi_h_arr[indices_to_plot_h],
                   color = "#fb8072", alpha = 0.8, label = "Adjusted Hill",
                   zorder = 10)
    if bootstrap_flag and k_h_star is not None:
        # Find the index of k_h_star in k_h_arr for plotting
        k_h_idx = np.argmin(np.abs(k_h_arr - k_h_star))
        axes[1,0].scatter([k_h_arr[k_h_idx]], [xi_h_arr[k_h_idx]],
                        color = "#fb8072", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Hill}="\
                            +str(np.round([xi_h_arr[k_h_idx]][0], decimals = 3))\
                            +r"$")
    axes[1,0].legend(loc = "best")

    
    axes[1,1].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[1,1].set_ylabel(r"Estimated $\hat{\xi}$", fontsize = 20) 
    axes[1,1].set_xscale("log")   
    
    # plot smooth Hill
    axes[1,1].plot(k_sh_arr[indices_to_plot_sh], xi_sh_arr[indices_to_plot_sh],
                   color = "#b3de69", alpha = 0.8, label = "Smooth Hill",
                   zorder = 10)
    # plot adjusted Hill
    indices_to_plot = np.where((k_h_arr <= max_k) & (k_h_arr >= min_k))
    axes[1,1].plot(k_h_arr[indices_to_plot_h], xi_h_arr[indices_to_plot_h],
                   color = "#fb8072", alpha = 0.8, label = "Adjusted Hill",
                   zorder = 10)
    if bootstrap_flag and k_h_star is not None:
        # Find the index of k_h_star in k_h_arr for plotting
        k_h_idx = np.argmin(np.abs(k_h_arr - k_h_star))
        axes[1,1].scatter([k_h_arr[k_h_idx]], [xi_h_arr[k_h_idx]],
                        color = "#fb8072", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Hill}="\
                            +str(np.round([xi_h_arr[k_h_idx]][0], decimals = 3))\
                            +r"$")
    axes[1,1].legend(loc = "best")

    axes[2,0].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[2,0].set_ylabel(r"Estimated $\hat{\xi}$", fontsize = 20)
    #plot Pickands
    min_k_index = (np.abs(k_p_arr - min_k)).argmin()
    max_k_index = (np.abs(k_p_arr - max_k)).argmin()
    if (xi_p_arr[min_k_index] <= -3 or xi_p_arr[min_k_index] >= 3):
        indices_to_plot_p = np.where((xi_p_arr <= 3) & (xi_p_arr >= -3))
    elif (xi_p_arr[max_k_index] <= -3 or xi_p_arr[max_k_index] >= 3):
        indices_to_plot_p = np.where((xi_p_arr <= 3) & (xi_p_arr >= -3))
    else:
        indices_to_plot_p = np.where((k_p_arr <= max_k) & (k_p_arr >= min_k))
    axes[2,0].plot(k_p_arr[indices_to_plot_p], xi_p_arr[indices_to_plot_p],
                   color = "#bc80bd", alpha = 0.8, label = "Pickands",
                   zorder = 10)
    #plot moments
    if (xi_m_arr[min_k-1] <= -3 or xi_m_arr[min_k-1] >= 3):
        indices_to_plot_m = np.where((xi_m_arr <= 3) & (xi_m_arr >= -3))
    elif (xi_m_arr[max_k-1] <= -3 or xi_m_arr[max_k-1] >= 3):
        indices_to_plot_m = np.where((xi_m_arr <= 3) & (xi_m_arr >= -3))
    else:
        indices_to_plot_m = np.where((k_m_arr <= max_k) & (k_m_arr >= min_k))
    
    axes[2,0].plot(k_m_arr[indices_to_plot_m], xi_m_arr[indices_to_plot_m],
                   color = "#8dd3c7", alpha = 0.8, label = "Moments",
                   zorder = 10)
    if bootstrap_flag and k_m_star is not None:
        # Find the index of k_m_star in k_m_arr for plotting
        k_m_idx = np.argmin(np.abs(k_m_arr - k_m_star))
        axes[2,0].scatter([k_m_arr[k_m_idx]], [xi_m_arr[k_m_idx]],
                        color = "#8dd3c7", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Moments}="\
                            +str(np.round([xi_m_arr[k_m_idx]][0], decimals = 3))\
                            +r"$")
    #plot kernel-type
    min_k_index = (np.abs(k_k_arr - min_k)).argmin()
    max_k_index = (np.abs(k_k_arr - max_k)).argmin()
    if (xi_k_arr[min_k_index] <= -3 or xi_k_arr[min_k_index] >= 3):
        indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    elif (xi_k_arr[max_k_index] <= -3 or xi_k_arr[max_k_index] >= 3):
        indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    else:
        indices_to_plot_k = list(range(min_k_index, max_k_index))
    #indices_to_plot_k = np.where((xi_k_arr <= 3) & (xi_k_arr >= -3))
    axes[2,0].plot(k_k_arr[indices_to_plot_k], xi_k_arr[indices_to_plot_k],
                   color = "#fdb462", alpha = 0.8, label = "Kernel",
                   zorder = 10)
    if bootstrap_flag and k_k1_star is not None:
        axes[2,0].scatter([k_k_arr[k_k1_star]], [xi_k_arr[k_k1_star]],
                        color = "#fdb462", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Kernel}="\
                            +str(np.round([xi_k_arr[k_k1_star]][0], decimals = 3))\
                            +r"$")
    axes[2,0].legend(loc = "best")
    # for clarity purposes, display only xi region between -1 and 1
    axes[2,0].set_ylim((-0.5,1.5))

    axes[2,1].set_xlabel(r"Number of Order Statistics $\kappa$", fontsize = 20)
    axes[2,1].set_ylabel(r"Estimated $\hat{\xi}$", fontsize = 20)
    axes[2,1].set_xscale("log")

    #plot Pickands
    axes[2,1].plot(k_p_arr[indices_to_plot_p], xi_p_arr[indices_to_plot_p],
                   color = "#bc80bd", alpha = 0.8, label = "Pickands",
                   zorder = 10)
    #plot moments
    axes[2,1].plot(k_m_arr[indices_to_plot_m], xi_m_arr[indices_to_plot_m],
                   color = "#8dd3c7", alpha = 0.8, label = "Moments",
                   zorder = 10)
    if bootstrap_flag and k_m_star is not None:
        # Find the index of k_m_star in k_m_arr for plotting
        k_m_idx = np.argmin(np.abs(k_m_arr - k_m_star))
        axes[2,1].scatter([k_m_arr[k_m_idx]], [xi_m_arr[k_m_idx]],
                        color = "#8dd3c7", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Moments}="\
                            +str(np.round([xi_m_arr[k_m_idx]][0], decimals = 3))\
                            +r"$")
    #plot kernel-type
    axes[2,1].plot(k_k_arr[indices_to_plot_k], xi_k_arr[indices_to_plot_k],
                   color = "#fdb462", alpha = 0.8, label = "Kernel",
                   zorder = 10)
    if bootstrap_flag and k_k1_star is not None:
        axes[2,1].scatter([k_k_arr[k_k1_star]], [xi_k_arr[k_k1_star]],
                        color = "#fdb462", marker = "*", s = 100,
                        edgecolor = "black", zorder = 20,
                        label = r"$\widehat{\xi}^{Kernel}="\
                            +str(np.round([xi_k_arr[k_k1_star]][0], decimals = 3))\
                            +r"$")
    # for clarity purposes, display only xi region between -1 and 1
    axes[2,1].set_ylim((-0.5,1.5))
    axes[2,1].legend(loc = "best")
    # If diagnostic plots are requested, they are now handled by make_diagnostic_plots()
    if diagn_plots and bootstrap_flag and verbose:
        print("Diagnostic plots will be created separately using plot_diagnostics()")
    
    fig.tight_layout(pad = 0.2)
    if output_file_path is not None:
        fig.savefig(output_file_path)
    
    return fig, axes

def make_diagnostic_plots(results, output_file_path=None,
                         hsteps=200, bootstrap_flag=True, verbose=False, noise_flag=True,
                         savedata=False):
    """
    Create diagnostic plots for tail index estimation.
    
    Parameters
    ----------
    results : dict
        Dictionary containing the results from fit_estimators.
    output_file_path : str or None, optional
        File path to which plots should be saved. If None, the figure is not saved.
    hsteps : int, optional
        Parameter controlling number of bandwidth steps of the kernel-type estimator, default is 200.
    bootstrap_flag : bool, optional
        Flag to switch on/off double-bootstrap procedure, default is True.
    verbose : bool, optional
        Flag controlling bootstrap verbosity, default is False.
    noise_flag : bool, optional
        Switch on/off uniform noise in range [-5*10^(-p), 5*10^(-p)] that is added to each
        data point. Used for integer-valued sequences with p = 1, default is True.
    savedata : bool, optional
        Flag to save data files in the directory with plots, default is False.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
    axes : numpy.ndarray or None
    """
    # Check if bootstrap is enabled
    if not bootstrap_flag:
        if verbose:
            print("Diagnostic plots require bootstrap to be enabled.")
        return None, None
    
    # Extract results for plotting
    ordered_data = results['ordered_data']
    if noise_flag and 'discrete_ordered_data' in results:
        discrete_ordered_data = results['discrete_ordered_data']
    
    # Hill results
    k_h_arr, xi_h_arr = results['hill']['k_arr_'], results['hill']['xi_arr_']
    if 'k_star_' in results['hill']:
        k_h_star = results['hill']['k_star_']
        xi_h_star = results['hill']['xi_star_']
        x1_h_arr = results['hill']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_h_amse = results['hill']['bootstrap_results_']['first_bootstrap_']['amse_']
        k1_h = results['hill']['bootstrap_results_']['first_bootstrap_']['k_min_']
        max_h_index1 = results['hill']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_h_arr = results['hill']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_h_amse = results['hill']['bootstrap_results_']['second_bootstrap_']['amse_']
        k2_h = results['hill']['bootstrap_results_']['second_bootstrap_']['k_min_']
        max_h_index2 = results['hill']['bootstrap_results_']['second_bootstrap_']['max_index_']
    else:
        k_h_star = None
        xi_h_star = None
        x1_h_arr, n1_h_amse, k1_h, max_h_index1 = None, None, None, None
        x2_h_arr, n2_h_amse, k2_h, max_h_index2 = None, None, None, None
    
    # Moments results
    k_m_arr, xi_m_arr = results['moments']['k_arr_'], results['moments']['xi_arr_']
    if 'k_star_' in results['moments']:
        k_m_star = results['moments']['k_star_']
        xi_m_star = results['moments']['xi_star_']
        x1_m_arr = results['moments']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_m_amse = results['moments']['bootstrap_results_']['first_bootstrap_']['amse_']
        k1_m = results['moments']['bootstrap_results_']['first_bootstrap_']['k_min_']
        max_m_index1 = results['moments']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_m_arr = results['moments']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_m_amse = results['moments']['bootstrap_results_']['second_bootstrap_']['amse_']
        k2_m = results['moments']['bootstrap_results_']['second_bootstrap_']['k_min_']
        max_m_index2 = results['moments']['bootstrap_results_']['second_bootstrap_']['max_index_']
    else:
        k_m_star = None
        xi_m_star = None
        x1_m_arr, n1_m_amse, k1_m, max_m_index1 = None, None, None, None
        x2_m_arr, n2_m_amse, k2_m, max_m_index2 = None, None, None, None
    
    # Kernel results
    k_k_arr, xi_k_arr = results['kernel']['k_arr_'], results['kernel']['xi_arr_']
    if 'k_star_' in results['kernel']:
        k_k_star = results['kernel']['k_star_']
        xi_k_star = results['kernel']['xi_star_']
        x1_k_arr = results['kernel']['bootstrap_results_']['first_bootstrap_']['x_arr_']
        n1_k_amse = results['kernel']['bootstrap_results_']['first_bootstrap_']['amse_']
        h1 = results['kernel']['bootstrap_results_']['first_bootstrap_']['h_min']
        max_k_index1 = results['kernel']['bootstrap_results_']['first_bootstrap_']['max_index_']
        x2_k_arr = results['kernel']['bootstrap_results_']['second_bootstrap_']['x_arr_']
        n2_k_amse = results['kernel']['bootstrap_results_']['second_bootstrap_']['amse_']
        h2 = results['kernel']['bootstrap_results_']['second_bootstrap_']['h_min']
        max_k_index2 = results['kernel']['bootstrap_results_']['second_bootstrap_']['max_index_']
        k_k1_star = results['kernel'].get('k_k1_star')
    else:
        k_k_star = None
        xi_k_star = None
        x1_k_arr, n1_k_amse, h1, max_k_index1 = None, None, None, None
        x2_k_arr, n2_k_amse, h2, max_k_index2 = None, None, None, None
        k_k1_star = None
    
    # Setup for saving data if needed
    if savedata and output_file_path is not None:
        output_dir = os.path.dirname(os.path.realpath(output_file_path))
        output_name = os.path.splitext(os.path.basename(output_file_path))[0]
    
    # Create diagnostic plots
    if verbose:
        print("Creating diagnostic plots...")
    
    fig_d, axes_d = plt.subplots(1, 3, figsize = (18, 6))

    # filter out boundary values using theta parameters for Hill
    if x1_h_arr is not None and x2_h_arr is not None:
        min_k1 = 2
        max_k1 = len(x1_h_arr) - 1
        min_k2 = 2
        max_k2 = len(x2_h_arr) - 1
    
    axes_d[0].set_yscale("log")
    axes_d[0].set_xscale("log")
    axes_d[1].set_xscale("log")
    axes_d[2].set_xscale("log")
    
    if n1_h_amse is not None and x1_h_arr is not None:
        n1_h_amse[np.where((n1_h_amse == np.inf) |\
                        (n1_h_amse == -np.inf))] = np.nan
        axes_d[0].set_ylim((0.1*np.nanmin(n1_h_amse[min_k1:max_k1]), 1.0))
    
    axes_d[0].set_xlabel("Fraction of Bootstrap Order Statistics",
                       fontsize = 20)
    axes_d[0].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
    axes_d[0].set_title("Adjusted Hill Estimator", fontsize = 20)
    
    # plot AMSE and corresponding minimum for Hill
    if x1_h_arr is not None and n1_h_amse is not None and k1_h is not None:
        axes_d[0].plot(x1_h_arr[min_k1:max_k1], n1_h_amse[min_k1:max_k1],
                    alpha = 0.5, lw = 1.5,
                    color = "#d55e00", label = r"$n_1$ samples")
        axes_d[0].scatter([k1_h], [n1_h_amse[int(len(x1_h_arr)*k1_h)-1]],
                        color = "#d55e00",
                        marker = 'o', edgecolor = "black", alpha = 0.5,
                        label = r"Min for $n_1$ sample")
    
    if x2_h_arr is not None and n2_h_amse is not None and k2_h is not None:
        axes_d[0].plot(x2_h_arr[min_k2:max_k2], n2_h_amse[min_k2:max_k2],
                    alpha = 0.5, lw = 1.5,
                    color = "#0072b2", label = r"$n_2$ samples")
        axes_d[0].scatter([k2_h], [n2_h_amse[int(len(x2_h_arr)*k2_h)-1]],
                        color = "#0072b2",
                        marker = 'o', edgecolor = "black", alpha = 0.5,
                        label = r"Min for $n_2$ sample")
    
    if max_h_index1 is not None and x1_h_arr is not None:
        axes_d[0].axvline(max_h_index1/float(len(x1_h_arr)), color = "#d55e00",
                        ls = '--', alpha = 0.5,
                        label = r"Minimization boundary for $n_1$ sample")
    
    if max_h_index2 is not None and x2_h_arr is not None:
        axes_d[0].axvline(max_h_index2/float(len(x2_h_arr)), color = "#0072b2",
                        ls = '--', alpha = 0.5,
                        label = r"Minimization boundary for $n_2$ sample")
    
    axes_d[0].legend(loc = "best")
    if savedata and output_file_path is not None:
        with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn1.dat"), "w") as f:
            for i in range(len(x1_h_arr[min_k1:max_k1])):
                f.write(str(x1_h_arr[min_k1:max_k1][i]) + " " + str(n1_h_amse[min_k1:max_k1][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn2.dat"), "w") as f:
            for i in range(len(x2_h_arr[min_k2:max_k2])):
                f.write(str(x2_h_arr[min_k2:max_k2][i]) + " " + str(n2_h_amse[min_k2:max_k2][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_adjhill_diagn_points.dat"), "w") as f:
            f.write("Min for n1 sample: "+str(k1_h)+" "+str(n1_h_amse[int(len(x1_h_arr)*k1_h)-1])+"\n")
            f.write("Min for n2 sample: "+str(k2_h)+" "+str(n2_h_amse[int(len(x2_h_arr)*k2_h)-1])+"\n")
            f.write("Minimization boundary for n1 sample: "+str(max_h_index1/float(len(x1_h_arr)))+"\n")
            f.write("Minimization boundary for n2 sample: "+str(max_h_index2/float(len(x2_h_arr)))+"\n")

    # filter out boundary values using theta parameters for moments
    min_k1 = 2
    max_k1 = len(x1_m_arr) - 1
    min_k2 = 2
    max_k2 = len(x2_m_arr) - 1
    n1_m_amse[np.where((n1_m_amse == np.inf) |\
                       (n1_m_amse == -np.inf))] = np.nan
    axes_d[1].set_yscale("log")
    axes_d[1].set_ylim((0.1*np.nanmin(n1_m_amse[min_k1:max_k1]), 1.0))
    axes_d[1].set_xlabel("Fraction of Bootstrap Order Statistics",
                       fontsize = 20)
    axes_d[1].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
    axes_d[1].set_title("Moments Estimator", fontsize = 20)
    # plot AMSE and corresponding minimum
    axes_d[1].plot(x1_m_arr[min_k1:max_k1], n1_m_amse[min_k1:max_k1],
                   alpha = 0.5, lw = 1.5,
                   color = "#d55e00", label = r"$n_1$ samples")
    axes_d[1].scatter([k1_m], [n1_m_amse[int(len(x1_m_arr)*k1_m)-1]],
                      color = "#d55e00",
                      marker = 'o', edgecolor = "black", alpha = 0.5,
                      label = r"Min for $n_1$ sample")
    axes_d[1].plot(x2_m_arr[min_k2:max_k2], n2_m_amse[min_k2:max_k2],
                   alpha = 0.5, lw = 1.5,
                   color = "#0072b2", label = r"$n_2$ samples")
    axes_d[1].scatter([k2_m], [n2_m_amse[int(len(x2_m_arr)*k2_m)-1]],
                      color = "#0072b2",
                      marker = 'o', edgecolor = "black", alpha = 0.5,
                      label = r"Min for $n_2$ sample")
    axes_d[1].axvline(max_m_index1/float(len(x1_m_arr)), color = "#d55e00",
                      ls = '--', alpha = 0.5,
                      label = r"Minimization boundary for $n_1$ sample")
    axes_d[1].axvline(max_m_index2/float(len(x2_m_arr)), color = "#0072b2",
                      ls = '--', alpha = 0.5,
                      label = r"Minimization boundary for $n_2$ sample")
    axes_d[1].legend(loc = "best")
    if savedata and output_file_path is not None:
        with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn1.dat"), "w") as f:
            for i in range(len(x1_m_arr[min_k1:max_k1])):
                f.write(str(x1_m_arr[min_k1:max_k1][i]) + " " + str(n1_m_amse[min_k1:max_k1][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn2.dat"), "w") as f:
            for i in range(len(x2_m_arr[min_k2:max_k2])):
                f.write(str(x2_m_arr[min_k2:max_k2][i]) + " " + str(n2_m_amse[min_k2:max_k2][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_mom_diagn_points.dat"), "w") as f:
            f.write("Min for n1 sample: "+str(k1_m)+" "+str(n1_m_amse[int(len(x1_m_arr)*k1_m)-1])+"\n")
            f.write("Min for n2 sample: "+str(k2_m)+" "+str(n2_m_amse[int(len(x2_m_arr)*k2_m)-1])+"\n")
            f.write("Minimization boundary for n1 sample: "+str(max_m_index1/float(len(x1_m_arr)))+"\n")
            f.write("Minimization boundary for n2 sample: "+str(max_m_index2/float(len(x2_m_arr)))+"\n")


    min_k1 = 2
    max_k1 = len(x1_k_arr)
    min_k2 = 2
    max_k2 = len(x2_k_arr)
    n1_k_amse[np.where((n1_k_amse == np.inf) |\
                       (n1_k_amse == -np.inf))] = np.nan
    axes_d[2].set_yscale("log")
    axes_d[2].set_ylim((0.1*np.nanmin(n1_k_amse[min_k1:max_k1]), 1.0))
    axes_d[2].set_xlabel("Fraction of Bootstrap Order Statistics",
                       fontsize = 20)
    axes_d[2].set_ylabel(r"$\langle AMSE \rangle$", fontsize = 20)
    axes_d[2].set_title("Kernel-type Estimator", fontsize = 20)
    # plot AMSE and corresponding minimum
    axes_d[2].plot(x1_k_arr[min_k1:max_k1], n1_k_amse[min_k1:max_k1],
                   alpha = 0.5, lw = 1.5,
                   color = "#d55e00", label = r"$n_1$ samples")
    axes_d[2].scatter([h1], [n1_k_amse[np.where(x1_k_arr == h1)][0]], color = "#d55e00",
                       marker = 'o', edgecolor = "black", alpha = 0.5,
                      label = r"Min for $n_1$ sample")
    # plot boundary of minimization
    axes_d[2].axvline(max_k_index1, color = "#d55e00",
                      ls = '--', alpha = 0.5,
                      label = r"Minimization boundary for $n_2$ sample")
    axes_d[2].plot(x2_k_arr[min_k2:max_k2], n2_k_amse[min_k2:max_k2],
                   alpha = 0.5, lw = 1.5,
                   color = "#0072b2", label = r"$n_2$ samples")
    axes_d[2].scatter([h2], [n2_k_amse[np.where(x2_k_arr == h2)][0]], color = "#0072b2",
                       marker = 'o', edgecolor = "black", alpha = 0.5,
                      label = r"Min for $n_2$ sample")
    axes_d[2].axvline(max_k_index2, color = "#0072b2",
                      ls = '--', alpha = 0.5,
                      label = r"Minimization boundary for $n_2$ sample")
    axes_d[2].legend(loc = "best")
    if savedata and output_file_path is not None:
        with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn1.dat"), "w") as f:
            for i in range(len(x1_k_arr[min_k1:max_k1])):
                f.write(str(x1_k_arr[min_k1:max_k1][i]) + " " + str(n1_k_amse[min_k1:max_k1][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn2.dat"), "w") as f:
            for i in range(len(x2_m_arr[min_k2:max_k2])):
                f.write(str(x2_k_arr[min_k2:max_k2][i]) + " " + str(n2_k_amse[min_k2:max_k2][i]) + "\n")
        with open(os.path.join(output_dir+"/"+output_name+"_kern_diagn_points.dat"), "w") as f:
            f.write("Min for n1 sample: "+str(h1)+" "+str(n1_k_amse[np.where(x1_k_arr == h1)][0])+"\n")
            f.write("Min for n2 sample: "+str(h2)+" "+str(n2_k_amse[np.where(x2_k_arr == h2)][0])+"\n")
            f.write("Minimization boundary for n1 sample: "+str(n1_k_amse[int(max_k_index1*hsteps)-1])+"\n")
            f.write("Minimization boundary for n2 sample: "+str(n2_k_amse[int(max_k_index2*hsteps)-1])+"\n")

    fig_d.tight_layout()
    if output_file_path is not None:
        diag_plots_path = os.path.dirname(os.path.realpath(output_file_path))+"/"+os.path.splitext(os.path.basename(output_file_path))[0]+"_diag.pdf"
        fig_d.savefig(diag_plots_path)
    
    return fig_d, axes_d
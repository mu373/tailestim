import time
import numpy as np

import logging
logging.basicConfig(level=logging.WARNING)

from .tail_methods import add_uniform_noise, get_distribution, get_ccdf

from .hill import HillEstimator
from .moments import MomentsEstimator
from .kernel import KernelTypeEstimator
from .pickands import PickandsEstimator
from .smooth_hill import SmoothHillEstimator

def fit_estimators(ordered_data, number_of_bins=30, r_smooth=2, alpha=0.6, hsteps=200,
                  bootstrap_flag=True, t_bootstrap=0.5, r_bootstrap=500, diagn_plots=False,
                  eps_stop=1.0, noise_flag=True, verbose=False, p_noise=1, base_seed=None):
    """
    Fit various tail estimators to the data at once.
    
    Parameters
    ----------
    ordered_data : numpy.ndarray
        Array for which tail index estimation is performed. Decreasing ordering is required.
    number_of_bins : int, default=30
        Number of log-bins for degree distribution.
    r_smooth : int, default=2
        Parameter controlling the width of smoothing window. Typically small value such as 2 or 3.
    alpha : float, default=0.6
        Parameter controlling the amount of "smoothing" for the kernel-type estimator. Should be greater than 0.5.
    hsteps : int, default=200
        Parameter controlling number of bandwidth steps of the kernel-type estimator.
    bootstrap_flag : bool, default=True
        Switch on/off double-bootstrap procedure.
    t_bootstrap : float, default=0.5
        Parameter controlling the size of the 2nd bootstrap. Defined from n2 = n*(t_bootstrap).
    r_bootstrap : int, default=500
        Number of bootstrap resamplings for the 1st and 2nd bootstraps.
    diagn_plots : bool, default=False
        Switch on/off generation of AMSE diagnostic plots.
    eps_stop : float, default=1.0
        Parameter controlling range of AMSE minimization. Defined as the fraction of order 
        statistics to consider during the AMSE minimization step.
    noise_flag : bool, default=True
        Switch on/off uniform noise in range [-5*10^(-p), 5*10^(-p)] that is added to each
        data point. Used for integer-valued sequences with p = 1.
    verbose : bool, default=False
        Flag controlling verbose logging.
    p_noise : int, default=1
        Parameter controlling noise amplitude.
    base_seed : int, optional
        Base random seed for reproducibility of bootstrap. Only used for methods with bootstrap.
        
    Returns
    -------
    dict
        A dictionary containing all estimation results including PDF, CCDF, and results from
        different estimators (Pickands, Hill, smooth Hill, moments, kernel-type).
    """
    results = {}
    
    # calculate log-binned PDF
    logging.debug("Calculating PDF...")
    t1 = time.time()
    x_pdf, y_pdf = get_distribution(ordered_data, number_of_bins=number_of_bins)
    t2 = time.time()
    logging.debug("Elapsed time(PDF):", t2-t1)
    results['pdf'] = {'x': x_pdf, 'y': y_pdf}

    # calculate CCDF
    logging.debug("Calculating CCDF...")
    t1 = time.time()
    x_ccdf, y_ccdf = get_ccdf(ordered_data)
    t2 = time.time()
    logging.debug("Elapsed time:", t2-t1)
    results['ccdf'] = {'x': x_ccdf, 'y': y_ccdf}

    # add noise if needed
    if noise_flag:
        original_discrete_data = ordered_data
        discrete_ordered_data = ordered_data.copy()
        discrete_ordered_data[::-1].sort()
        ordered_data = add_uniform_noise(ordered_data, p=p_noise, base_seed=base_seed)
    ordered_data[::-1].sort()
    results['ordered_data'] = ordered_data
    if noise_flag:
        results['discrete_ordered_data'] = discrete_ordered_data

    # perform Pickands estimation
    logging.debug("Calculating Pickands...")
    t1 = time.time()
    pickands = PickandsEstimator()
    pickands.fit(ordered_data)
    pickands_result = pickands.get_result()
    k_p_arr, xi_p_arr = pickands_result.k_arr_, pickands_result.xi_arr_
    t2 = time.time()
    logging.debug("Elapsed time (Pickands):", t2-t1)
    results['pickands'] = {'k_arr_': k_p_arr, 'xi_arr_': xi_p_arr}

    # perform smooth Hill estimation
    logging.debug("Calculating smooth Hill...")
    t1 = time.time()
    smooth_hill = SmoothHillEstimator(r_smooth=r_smooth)
    smooth_hill.fit(ordered_data)
    smooth_hill_result = smooth_hill.get_result()
    k_sh_arr, xi_sh_arr = smooth_hill_result.k_arr_, smooth_hill_result.xi_arr_
    t2 = time.time()
    logging.debug("Elapsed time (smooth Hill):", t2-t1)
    results['smooth_hill'] = {'k_arr_': k_sh_arr, 'xi_arr_': xi_sh_arr}

    # perform adjusted Hill estimation
    logging.debug("Calculating adjusted Hill...")
    t1 = time.time()
    hill = HillEstimator(
        bootstrap=bootstrap_flag,
        t_bootstrap=t_bootstrap,
        r_bootstrap=r_bootstrap,
        diagn_plots=diagn_plots,
        eps_stop=eps_stop,
        verbose=verbose,
        base_seed=base_seed
    )
    hill.fit(ordered_data)
    hill_result = hill.get_result()
    k_h_arr = hill_result.k_arr_
    xi_h_arr = hill_result.xi_arr_
    
    results['hill'] = {'k_arr_': k_h_arr, 'xi_arr_': xi_h_arr}
    
    if bootstrap_flag and hasattr(hill_result, 'k_star_'):
        k_h_star = hill_result.k_star_
        xi_h_star = hill_result.xi_star_
        x1_h_arr = hill_result.bootstrap_results_.first_bootstrap_.x_arr_
        n1_h_amse = hill_result.bootstrap_results_.first_bootstrap_.amse_
        k1_h = hill_result.bootstrap_results_.first_bootstrap_.k_min_
        max_h_index1 = hill_result.bootstrap_results_.first_bootstrap_.max_index_
        x2_h_arr = hill_result.bootstrap_results_.second_bootstrap_.x_arr_
        n2_h_amse = hill_result.bootstrap_results_.second_bootstrap_.amse_
        k2_h = hill_result.bootstrap_results_.second_bootstrap_.k_min_
        max_h_index2 = hill_result.bootstrap_results_.second_bootstrap_.max_index_
        
        results['hill'].update({
            'k_star_': k_h_star,
            'xi_star_': xi_h_star,
            'bootstrap_results_': {
                'first_bootstrap_': {
                    'x_arr_': x1_h_arr,
                    'amse_': n1_h_amse,
                    'k_min_': k1_h,
                    'max_index_': max_h_index1
                },
                'second_bootstrap_': {
                    'x_arr_': x2_h_arr,
                    'amse_': n2_h_amse,
                    'k_min_': k2_h,
                    'max_index_': max_h_index2
                }
            }
        })
    
    t2 = time.time()
    if verbose:
        logging.debug("Elapsed time (Hill):", t2-t1)

    # perform moments estimation
    if verbose:
        logging.debug("Calculating moments...")
    t1 = time.time()
    moments = MomentsEstimator(
        bootstrap=bootstrap_flag,
        t_bootstrap=t_bootstrap,
        r_bootstrap=r_bootstrap,
        diagn_plots=diagn_plots,
        eps_stop=eps_stop,
        verbose=verbose,
        base_seed=base_seed
    )
    moments.fit(ordered_data)
    moments_result = moments.get_result()
    k_m_arr = moments_result.k_arr_
    xi_m_arr = moments_result.xi_arr_
    
    results['moments'] = {'k_arr_': k_m_arr, 'xi_arr_': xi_m_arr}
    
    if bootstrap_flag and hasattr(moments_result, 'k_star_'):
        k_m_star = moments_result.k_star_
        xi_m_star = moments_result.xi_star_
        x1_m_arr = moments_result.bootstrap_results_.first_bootstrap_.x_arr_
        n1_m_amse = moments_result.bootstrap_results_.first_bootstrap_.amse_
        k1_m = moments_result.bootstrap_results_.first_bootstrap_.k_min_
        max_m_index1 = moments_result.bootstrap_results_.first_bootstrap_.max_index_
        x2_m_arr = moments_result.bootstrap_results_.second_bootstrap_.x_arr_
        n2_m_amse = moments_result.bootstrap_results_.second_bootstrap_.amse_
        k2_m = moments_result.bootstrap_results_.second_bootstrap_.k_min_
        max_m_index2 = moments_result.bootstrap_results_.second_bootstrap_.max_index_
        
        results['moments'].update({
            'k_star_': k_m_star,
            'xi_star_': xi_m_star,
            'bootstrap_results_': {
                'first_bootstrap_': {
                    'x_arr_': x1_m_arr,
                    'amse_': n1_m_amse,
                    'k_min_': k1_m,
                    'max_index_': max_m_index1
                },
                'second_bootstrap_': {
                    'x_arr_': x2_m_arr,
                    'amse_': n2_m_amse,
                    'k_min_': k2_m,
                    'max_index_': max_m_index2
                }
            }
        })
    
    t2 = time.time()
    if verbose:
        logging.debug("Elapsed time (moments):", t2-t1)

    # perform kernel-type estimation
    if verbose:
        logging.debug("Calculating kernel-type...")
    t1 = time.time()
    kernel = KernelTypeEstimator(
        hsteps=hsteps,
        alpha=alpha,
        bootstrap=bootstrap_flag,
        t_bootstrap=t_bootstrap,
        r_bootstrap=r_bootstrap,
        diagn_plots=diagn_plots,
        eps_stop=eps_stop,
        verbose=verbose,
        base_seed=base_seed
    )
    kernel.fit(ordered_data)
    kernel_result = kernel.get_result()
    k_k_arr = kernel_result.k_arr_
    xi_k_arr = kernel_result.xi_arr_
    
    results['kernel'] = {'k_arr_': k_k_arr, 'xi_arr_': xi_k_arr}
    
    if bootstrap_flag and hasattr(kernel_result, 'k_star_'):
        k_k_star = kernel_result.k_star_
        xi_k_star = kernel_result.xi_star_
        x1_k_arr = kernel_result.bootstrap_results_.first_bootstrap_.x_arr_
        n1_k_amse = kernel_result.bootstrap_results_.first_bootstrap_.amse_
        h1 = kernel_result.bootstrap_results_.first_bootstrap_.h_min_
        max_k_index1 = kernel_result.bootstrap_results_.first_bootstrap_.max_index_
        x2_k_arr = kernel_result.bootstrap_results_.second_bootstrap_.x_arr_
        n2_k_amse = kernel_result.bootstrap_results_.second_bootstrap_.amse_
        h2 = kernel_result.bootstrap_results_.second_bootstrap_.h_min_
        max_k_index2 = kernel_result.bootstrap_results_.second_bootstrap_.max_index_
        
        results['kernel'].update({
            'k_star_': k_k_star,
            'xi_star_': xi_k_star,
            'bootstrap_results_': {
                'first_bootstrap_': {
                    'x_arr_': x1_k_arr,
                    'amse_': n1_k_amse,
                    'h_min': h1,
                    'max_index_': max_k_index1
                },
                'second_bootstrap_': {
                    'x_arr_': x2_k_arr,
                    'amse_': n2_k_amse,
                    'h_min': h2,
                    'max_index_': max_k_index2
                }
            }
        })
        
        # Find the index of k_k_star in k_k_arr for plotting
        if k_k_star is not None:
            k_k1_star = np.argmin(np.abs(k_k_arr - k_k_star))
            results['kernel']['k_k1_star'] = k_k1_star
    
    t2 = time.time()
    if verbose:
        logging.debug("Elapsed time (kernel-type):", t2-t1)
        
    return results
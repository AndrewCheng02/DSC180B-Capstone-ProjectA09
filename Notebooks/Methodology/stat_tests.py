import numpy as np 
from scipy.stats import t
from nilearn import plotting

def get_tstats(group1, group2, pval=True): 
    """
    Finds t-statistics for a two-sample t-test given the difference of means between
    group 1 and group 2 (assumes unequal variances).

    * note that this is for testing many hypotheses simultaneously.
    
    args:
        group 1 (arrayLike)          - sample values from group 1
        group 2 (arrayLike)          - sample values from group 2


    returns:
        t_stats (2D np.array) - t-statistics for the given pairwise correlations.
        g1_sqse (float)       - squared standard error of group 1
        g2_sqse (float)       - squared standard error of group 2
    """
    # calculate the t-statistic for each pairwise correlation value 
    # STEP 1: Calculate the estimated overall standard error for both groups  
    def get_squared_se(group_netmats):
        n = len(group_netmats) # sample size
        ss = lambda netmats : np.sum((netmats - np.mean(netmats, axis=0))**2, axis=0) # sum of squared difference from the mean
        var = lambda netmats: ss(netmats) / (n - 1) # variance 
        return var(group_netmats) / n
    
    g1_sqse = get_squared_se(group1) # Group 1 squared standard error
    g2_sqse = get_squared_se(group2) # Group 2 squared standard error
    
    overall_se = np.sqrt(g1_sqse + g2_sqse) # Estimated overall standard error

    # STEP 2: Calculate the t-statistic
    diff_means = np.mean(group1, axis=0) - np.mean(group2, axis=0)
    t_stats = diff_means / overall_se

    # STEP 3: Get p-values (optional)
    if pval:
        dof = group1.shape[0] + group2.shape[0] # n1 + n2
        pvals = t.sf(abs(t_stats), df=dof)*2 # two-sided t-test
        return t_stats, [g1_sqse, g2_sqse], pvals

    else:
        return t_stats, [g1_sqse, g2_sqse]
    
def get_sig_regions(regions, p_values, alpha, title="Statistically Signficant Pairwise Correlations Between Males and Females"):    
    """
    Returns a matrix of statistically significant values.

    * note that this is for testing many hypotheses simultaneously.
    
    args:
        regions (2D np.array) - square symmetric ixj matrix containing the observed values.
        p_values (2D np.array) - square symmetric ixj matrix containing the p-values of the observed values.
        alpha (float) - the significance level

    returns:
        significant_regions (2D np.array) - square symmetric ixj matrix containing the observed values with a p-value < alpha.
                                            all other values are set to 0.
    """
    significant_regions = np.where(p_values < alpha, regions, 0)

    values_sd = np.std(significant_regions)
    values_mean = np.round(np.mean(significant_regions))

    vmax = values_mean + 3 * values_sd # capture 99.7 percent of the dataset, values more than 3 sd from mean casted to extreme colors
    vmin = values_mean - 3 * values_sd

    plotting.plot_matrix(
        significant_regions, colorbar=True, vmax=vmax, vmin=vmin, title=title)
    
    return significant_regions
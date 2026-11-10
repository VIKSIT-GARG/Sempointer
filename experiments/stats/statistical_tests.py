import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional

def bootstrap_ci(
    values: List[float],
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Returns (point_estimate, lower_ci, upper_ci)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    values_arr = np.array(values)
    point_est = statistic(values_arr)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        boot_stats.append(statistic(sample))
        
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(point_est), float(lower), float(upper)

def bootstrap_ci_batch(
    metrics_dict: Dict[str, List[float]],
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict[str, Dict]:
    results = {}
    for name, vals in metrics_dict.items():
        if not vals:
            continue
        est, lower, upper = bootstrap_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha)
        results[name] = {'mean': est, 'ci_lower': lower, 'ci_upper': upper}
    return results

def mcnemar_test(
    system_a_correct: List[int],
    system_b_correct: List[int],
    correction: bool = True
) -> Dict:
    b = sum(1 for a, bb in zip(system_a_correct, system_b_correct) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(system_a_correct, system_b_correct) if a == 0 and bb == 1)
    
    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False, 'odds_ratio': 1.0}
        
    if correction:
        stat = ((abs(b - c) - 1) ** 2) / (b + c)
    else:
        stat = ((b - c) ** 2) / (b + c)
        
    p_val = 1 - stats.chi2.cdf(stat, 1)
    odds_ratio = b / c if c > 0 else float('inf')
    
    return {
        'statistic': float(stat),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
        'odds_ratio': float(odds_ratio)
    }

def wilcoxon_test(
    values_a: List[float],
    values_b: List[float],
    alternative: str = 'two-sided'
) -> Dict:
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False, 'cohens_d': 0.0}
        
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if all(d == 0 for d in diffs):
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False, 'cohens_d': 0.0}
        
    stat, p_val = stats.wilcoxon(values_a, values_b, alternative=alternative)
    d = cohens_d(values_a, values_b)
    
    return {
        'statistic': float(stat),
        'p_value': float(p_val),
        'significant': bool(p_val < 0.05),
        'cohens_d': float(d)
    }

def cohens_d(values_a: List[float], values_b: List[float]) -> float:
    n1, n2 = len(values_a), len(values_b)
    if n1 == 0 or n2 == 0:
        return 0.0
    var1 = np.var(values_a, ddof=1)
    var2 = np.var(values_b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(values_a) - np.mean(values_b)) / pooled_sd

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(p_values)
    if m == 0:
        return []
    adj_alpha = alpha / m
    return [p < adj_alpha for p in p_values]

def compute_all_significance(
    sempointer_results: List[float],
    baseline_results: List[float],
    metric_name: str,
    binary: bool = False
) -> Dict:
    result = {'metric_name': metric_name}
    
    est_sem, l_sem, u_sem = bootstrap_ci(sempointer_results)
    est_base, l_base, u_base = bootstrap_ci(baseline_results)
    
    result['sempointer'] = {'mean': est_sem, 'ci_lower': l_sem, 'ci_upper': u_sem}
    result['baseline'] = {'mean': est_base, 'ci_lower': l_base, 'ci_upper': u_base}
    
    if binary:
        sem_ints = [int(v) for v in sempointer_results]
        base_ints = [int(v) for v in baseline_results]
        test_res = mcnemar_test(sem_ints, base_ints)
    else:
        test_res = wilcoxon_test(sempointer_results, baseline_results)
        
    result['test'] = test_res
    return result

def power_analysis(
    effect_size: float = 0.1,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    try:
        from statsmodels.stats.power import TTestIndPower
    except ImportError:
        return int(2 * ((stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)) / effect_size) ** 2)
        
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(n))

def fit_log_degradation(
    N_values: List[int],
    recall_values: List[float]
) -> Dict:
    if len(N_values) < 2:
        return {'beta': 0.0, 'r_squared': 0.0, 'fitted_values': []}
        
    log_N = np.log10(N_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, recall_values)
    fitted = intercept + slope * log_N
    
    return {
        'beta': float(-slope),
        'r_squared': float(r_value ** 2),
        'fitted_values': fitted.tolist()
    }

def fit_linear(
    x_values: List[float],
    y_values: List[float]
) -> Dict:
    if len(x_values) < 2:
        return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0, 'p_value': 1.0}
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value)
    }

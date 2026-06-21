"""
Statistical utilities for CRITIQ-BIAS v2.0 metrics.

Provides:
- Bootstrap confidence intervals
- Effect sizes (Cohen's d)
- Paired comparisons
- ANOVA helpers
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class StatResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float | None = None
    significant: bool = False
    method: str = ""


class StatisticalTests:
    """Statistical testing utilities for bias metrics."""
    
    @staticmethod
    def bootstrap_ci(
        data: list[float],
        statistic_func: Callable[[list[float]], float] = np.mean,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Sample data
            statistic_func: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (default 0.05 for 95% CI)
            seed: Random seed for reproducibility
            
        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        if len(data) == 0:
            return (0.0, 0.0, 0.0)
        
        rng = np.random.RandomState(seed)
        data_arr = np.array(data)
        n = len(data_arr)
        
        # Point estimate
        point_estimate = float(statistic_func(data_arr))
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data_arr, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        # Percentile confidence interval
        ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
        
        return (point_estimate, ci_lower, ci_upper)
    
    @staticmethod
    def bootstrap_mfi(
        target_scores: list[float],
        other_scores: list[float],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """
        Bootstrap confidence interval for MFI ratio.
        
        MFI = mean(target_scores) / mean(other_scores)
        
        Returns:
            (mfi_estimate, ci_lower, ci_upper)
        """
        if len(target_scores) == 0 or len(other_scores) == 0:
            return (0.0, 0.0, 0.0)
        
        rng = np.random.RandomState(seed)
        target_arr = np.array(target_scores)
        other_arr = np.array(other_scores)
        
        mfi_samples = []
        for _ in range(n_bootstrap):
            t_sample = rng.choice(target_arr, size=len(target_arr), replace=True)
            o_sample = rng.choice(other_arr, size=len(other_arr), replace=True)
            
            other_mean = np.mean(o_sample)
            if other_mean > 0:
                mfi_samples.append(np.mean(t_sample) / other_mean)
        
        if len(mfi_samples) == 0:
            return (0.0, 0.0, 0.0)
        
        point_estimate = float(np.mean(mfi_samples))
        ci_lower = float(np.percentile(mfi_samples, 2.5))
        ci_upper = float(np.percentile(mfi_samples, 97.5))
        
        return (point_estimate, ci_lower, ci_upper)
    
    @staticmethod
    def cohens_d(group1: list[float], group2: list[float]) -> float:
        """
        Compute Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - |d| < 0.5: small
        - |d| < 0.8: medium  
        - |d| >= 0.8: large
        """
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return float((mean1 - mean2) / pooled_std)
    
    @staticmethod
    def paired_difference_test(
        scores_condition1: list[float],
        scores_condition2: list[float],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> StatResult:
        """
        Bootstrap test for paired differences (e.g., visible vs blind scores).
        
        Tests H0: mean(condition1) - mean(condition2) = 0
        """
        if len(scores_condition1) != len(scores_condition2):
            raise ValueError("Paired test requires equal-length samples")
        
        differences = [a - b for a, b in zip(scores_condition1, scores_condition2)]
        mean_diff, ci_lower, ci_upper = StatisticalTests.bootstrap_ci(
            differences, np.mean, n_bootstrap, seed=seed
        )
        
        # Significant if CI excludes 0
        significant = ci_lower > 0 or ci_upper < 0
        
        # Effect size for paired difference
        std_diff = np.std(differences, ddof=1) if len(differences) > 1 else 1.0
        effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
        
        return StatResult(
            statistic=mean_diff,
            p_value=0.0,  # Bootstrap doesn't give exact p-value
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            effect_size=float(effect_size),
            significant=significant,
            method="bootstrap_paired",
        )
    
    @staticmethod
    def interpret_effect_size(d: float) -> str:
        """Human-readable interpretation of Cohen's d."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def mfi_significant(ci_lower: float, ci_upper: float) -> str:
        """
        Interpret MFI result based on confidence interval.

        - If CI excludes 1.0: statistically significant bias
        - If CI > 1.0: favors this creator
        - If CI < 1.0: disfavors this creator
        """
        if ci_lower > 1.0:
            return "significantly_favored"
        elif ci_upper < 1.0:
            return "significantly_disfavored"
        else:
            return "no_significant_bias"

    @staticmethod
    def one_way_anova(groups: list[list[float]]) -> StatResult:
        """
        One-way ANOVA across multiple groups.

        Returns F-statistic, p-value, and eta-squared effect size.
        """
        from scipy import stats

        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            return StatResult(
                statistic=0.0, p_value=1.0, ci_lower=0.0, ci_upper=0.0,
                method="one_way_anova",
            )

        group_means = [float(np.mean(g)) for g in groups]
        if len({round(m, 10) for m in group_means}) == 1:
            return StatResult(
                statistic=0.0,
                p_value=1.0,
                ci_lower=0.0,
                ci_upper=0.0,
                effect_size=0.0,
                significant=False,
                method="one_way_anova",
            )

        f_stat, p_value = stats.f_oneway(*groups)

        if np.isnan(f_stat) or np.isnan(p_value):
            return StatResult(
                statistic=0.0,
                p_value=1.0,
                ci_lower=0.0,
                ci_upper=0.0,
                effect_size=0.0,
                significant=False,
                method="one_way_anova",
            )

        # Eta-squared effect size
        all_values = [v for g in groups for v in g]
        grand_mean = np.mean(all_values)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum((v - grand_mean) ** 2 for v in all_values)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        return StatResult(
            statistic=float(f_stat),
            p_value=float(p_value),
            ci_lower=0.0,
            ci_upper=0.0,
            effect_size=float(eta_squared),
            significant=p_value < 0.05,
            method="one_way_anova",
        )

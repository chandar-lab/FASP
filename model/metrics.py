import torch
import numpy as np
from scipy.stats import anderson_ksamp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=-1).to(device)

def calculate_significance(scores):
    different_distributions = (
        anderson_ksamp(list(scores.values())).significance_level < 0.05
    )
    return different_distributions

def evaluate_fairness_disparity(domain_results_pd, scores):
    mean = {
        group: np.mean(scores) for group, scores in scores.items()
    }
    vars = {
        group: np.var(scores) for group, scores in scores.items()
    }
    range_mean = np.ptp(list(mean.values()))
    range_vars = np.ptp(list(vars.values()))
    ad_test_result = calculate_significance(scores)
    print(f"The averages are {mean}")
    print(f"The variance are {vars}")
    print(
        f"We can reject the null hypothesis for AD test \
           (null hypothesis: the populations from which two or more groups \
           of data were drawn are identical): {ad_test_result}"
    )
    if range_mean > 0.1:
        min_group = min(mean, key=scores.get)
        max_group = max(mean, key=scores.get)
        # permutation_test in Colab cannot be run since Colab currently runs on
        # python version 3.7.14, and scipy 1.8.0 requires higher version.
        # p_test_results  = permutation_test(
        #     (scores[min_group], scores[max_group]),
        #     mean_difference,
        #     n_resamples=np.inf,
        #     alternative='less',
        # )
        print(
            f"The min and max mean is group {min_group} \
            with {mean[min_group]} and \
            {max_group} with {mean[max_group]}"
        )
        # print(f"The permutation test results are {p_test_results}")

    if range_vars > 0.1:
        min_group = min(vars, key=scores.get)
        max_group = max(vars, key=scores.get)
        # p_test_results = permutation_test(
        #     (scores[min_group], scores[max_group]),
        #     var_difference,
        #     n_resamples=np.inf,
        #     alternative="less",
        # )
        print(
            f"The min and max variance is group {min_group} \
            with {vars[min_group]} and \
            {max_group} with {vars[max_group]}"
        )
        # print(f"The permutation test results are {p_test_results}")
    return {"mean": mean, "var": vars, "raw": scores}

"""Performance indicators for multi-objective RL algorithms.

We mostly rely on pymoo for the computation of axiomatic indicators (HV and IGD), but some are customly made.
"""

from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

import numpy as np
from typing import List, Dict

# This helper class is needed for the constraint satisfaction metric
class ReturnConstraint:
    """Helper class to check if a multi-objective return satisfies a set of linear constraints."""
    def __init__(self, weights: np.ndarray, thresholds: np.ndarray):
        self.weights = weights
        self.thresholds = thresholds

    def __call__(self, point: np.ndarray) -> bool:
        # Checks if `w^T * point >= t` for all w and t.
        # The original code's `(x, y)` implies a 2D point. This generalizes it.
        return all(np.dot(w, point) >= t for w, t in zip(self.weights, self.thresholds))

def variance_objective_metric(
    front: List[np.ndarray],
    reward_dim: int,
    n_samples: int = 100
) -> float:
    """
    Calculates the variance objective metric.

    This metric rewards higher mean returns and penalizes higher variance.
    """
    all_results = []
    # Generalize weights for any reward_dim. We need 2*reward_dim weights:
    # one set for the mean (positive contribution) and one for the variance (negative contribution).
    weights = np.random.uniform(0, 1, size=(n_samples, 2 * reward_dim))
    weights = weights / weights.sum(axis=1, keepdims=True)

    for i in range(n_samples):
        results_for_weight = []
        for policy_returns in front:
            # Ensure policy_returns is a 2D array
            if policy_returns.ndim == 1:
                policy_returns = np.expand_dims(policy_returns, axis=0)
            
            curr_mean = policy_returns.mean(axis=0)
            curr_var = policy_returns.std(axis=0)
            
            mean_weights = weights[i, :reward_dim]
            var_weights = weights[i, reward_dim:]
            
            score = np.sum(curr_mean * mean_weights) - np.sum(curr_var * var_weights)
            results_for_weight.append(score)
        all_results.append(np.max(results_for_weight))

    return np.mean(all_results)

def constraint_satisfaction_metric(
    front: List[np.ndarray],
    reward_dim: int,
    return_bounds: Dict[str, np.ndarray],
    n_samples: int = 100
) -> float:
    """
    Calculates the constraint satisfaction metric.

    This metric measures the ability of the policy set to satisfy randomly generated
    linear constraints on the return space.
    """
    constraints = []
    min_return, max_return = return_bounds['minimum'], return_bounds['maximum']
    
    # Generate random constraints, making the function self-contained
    for _ in range(n_samples):
        # Original code used 1 or 2 constraints. We'll stick to that for consistency.
        num_sub_constraints = np.random.randint(1, 3)
        
        # Generate weights that sum to 1
        constraint_weights = np.random.dirichlet(np.ones(reward_dim), size=num_sub_constraints)
        
        # Generate a random threshold for each sub-constraint
        thresholds = []
        for w in constraint_weights:
            min_val = np.dot(w, min_return)
            max_val = np.dot(w, max_return)
            thresholds.append(np.random.uniform(min_val, max_val))
        
        constraints.append(ReturnConstraint(constraint_weights, np.array(thresholds)))

    # Calculate satisfaction
    avg_max_prob = []
    for constr in constraints:
        probs = []
        for policy_returns in front:
            if policy_returns.ndim == 1:
                policy_returns = np.expand_dims(policy_returns, axis=0)
            # Probability is the mean number of times the constraint is met
            prob = np.mean([constr(point) for point in policy_returns])
            probs.append(prob)
        avg_max_prob.append(np.max(probs))

    return np.mean(avg_max_prob)
    
def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

def spacing(front: List[np.ndarray]) -> float:
    """Spacing metric - measures uniformity of distribution along the front."""
    if len(front) < 2:
        return 0.0
    
    front_array = np.array(front)
    distances = []
    
    for i, point in enumerate(front_array):
        # Find distances to all other points
        other_points = np.delete(front_array, i, axis=0)
        min_dist = np.min(np.linalg.norm(other_points - point, axis=1))
        distances.append(min_dist)
    
    # Standard deviation of distances
    return np.std(distances)
    
def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """Inverted generational distance metric. Requires to know the optimal front.

    Args:
        known_front: known pareto front for the problem
        current_estimate: current pareto front

    Return:
        a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    (!) This metric only considers the points from the PF identified by the algorithm, not the full objective space.
    Therefore, it is misleading (e.g. learning only one point is considered good) and we recommend not using it when comparing algorithms.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def cardinality(front: List[np.ndarray]) -> float:
    """Cardinality Metric.

    Cardinality of the Pareto front approximation.

    Args:
        front: current pareto front to compute the cardinality on

    Returns:
        float: cardinality metric
    """
    return len(front)


def maximum_utility_loss(
    front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in range(len(max_scalarized_values))]
    return np.max(utility_losses)

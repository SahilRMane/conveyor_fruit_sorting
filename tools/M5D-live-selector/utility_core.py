"""Image features, a small ridge model, and a deadline guard."""

import cv2
import numpy as np


VIEWS = ('inspect_left', 'inspect_right')
RIDGE_PENALTY = 4.0
WRONG_CLASS_COST = 2.0
TIME_COST_PER_S = 0.01


def assess_features(rgb_crop, confidence):
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).astype(float) / 255.
    return np.r_[patch.ravel(), confidence]


def fit(features, payoffs):
    features, payoffs = np.asarray(features), np.asarray(payoffs)
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .02)
    z = (features - center) / scale
    bias = payoffs.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + RIDGE_PENALTY * np.eye(z.shape[1]), z.T @ (payoffs - bias))
    return dict(center=center, scale=scale, bias=bias, weights=weights)


def predict(model, features):
    z = (np.asarray(features) - model['center']) / model['scale']
    return np.clip(z @ model['weights'] + model['bias'], -WRONG_CLASS_COST, 1.)


def choose_view(predicted_payoffs, view_costs, remaining_s, pick_reserve_s):
    """Return a view only if it has positive utility and positive slack."""
    best = None
    best_utility = 0.
    for view, payoff in zip(VIEWS, predicted_payoffs, strict=True):
        cost = view_costs[view]
        utility = payoff - TIME_COST_PER_S * cost
        if remaining_s > cost + pick_reserve_s and utility > best_utility:
            best, best_utility = view, utility
    return best

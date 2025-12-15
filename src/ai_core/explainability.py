import shap
import matplotlib.pyplot as plt
import pandas as pd


def compute_shap_values(model, X):
    """
    Compute SHAP values for tree-based models.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_global_importance(shap_values, X_sample, title="Feature Importance"):
    """
    Global SHAP importance plot.
    """
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )
    plt.title(title)
    return plt


def plot_local_explanation(explainer, shap_values, X_sample, index=0):
    """
    Local explanation for a single prediction.
    """
    return shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X_sample.iloc[index],
        matplotlib=True,
        show=False
    )

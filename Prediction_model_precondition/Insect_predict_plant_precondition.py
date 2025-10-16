import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import NotFittedError


class CurveExpertEmulator(BaseEstimator, RegressorMixin):
    def __init__(self, progress_bar=False):
        self.progress_bar = progress_bar
        self.models = {
            # Basic models
            "linear": lambda x, a, b: a + b * x,
            "quadratic": lambda x, a, b, c: a + b * x + c * x ** 2,
            "cubic": lambda x, a, b, c, d: a + b * x + c * x ** 2 + d * x ** 3,
            "power": lambda x, a, b: a * x ** b,
            "geometric": lambda x, a, b: a * x ** (b * x),
            "logarithmic": lambda x, a, b: a + b * np.log(x + 1e-10),
            "exponential": lambda x, a, b: a * np.exp(b * x),

            # Complex models
            "bleasdale": lambda x, a, b, c: (a + b * x) ** (-1 / c),
            "richards": lambda x, a, b, c, d: a / (1 + np.exp(b - c * x)) ** (1 / d),
            "weibull": lambda x, a, b, c, d: a - b * np.exp(-c * x ** d),
            "hoerl": lambda x, a, b, c: a * (b ** x) * (x ** c),
            "shifted_power": lambda x, a, b, c: a * (x - b) ** c,
            "vapor_pressure": lambda x, a, b, c: np.exp(a + b / (x + 1e-10) + c * np.log(x + 1e-10)),
            "heat_capacity": lambda x, a, b, c: a + b * x + c / (x ** 2 + 1e-10),

            # Additional models
            "rational": lambda x, a, b, c, d: (a + b * x) / (1 + c * x + d * x ** 2),
            "modified_hoerl": lambda x, a, b, c: a * b ** (1 / (x + 1e-10)) * (x ** c),
            "sinusoidal": lambda x, a, b, c, d: a + b * np.cos(c * x + d),
            "gompertz": lambda x, a, b, c: a * np.exp(-np.exp(b - c * x)),
            "saturation_growth": lambda x, a, b: a * x / (b + x),
            "gaussian": lambda x, a, b, c: a * np.exp(-((b - x) ** 2) / (2 * c ** 2)),
            "logistic": lambda x, a, b, c: a / (1 + b * np.exp(-c * x)),
            "modified_geometric": lambda x, a, b: a * x ** (b / (x + 1e-10)),
            "mmf": lambda x, a, b, c, d: (a * b + c * x ** d) / (b + x ** d),
            "modified_power": lambda x, a, b: a * (b ** x),
            "modified_exponential": lambda x, a, b: a * np.exp(b / (x + 1e-10))
        }

        self.bounds = {
            "linear": ([-np.inf] * 2, [np.inf] * 2),
            "quadratic": ([-np.inf] * 3, [np.inf] * 3),
            "cubic": ([-np.inf] * 4, [np.inf] * 4),
            "power": ([0, -np.inf], [np.inf, np.inf]),
            "geometric": ([0, 0], [np.inf, np.inf]),
            "logarithmic": ([-np.inf, -np.inf], [np.inf, np.inf]),
            "exponential": ([0, -np.inf], [np.inf, np.inf]),
            "bleasdale": ([0] * 3, [np.inf] * 3),
            "richards": ([0] * 4, [np.inf] * 4),
            "weibull": ([-np.inf] * 4, [np.inf] * 4),
            "hoerl": ([0, 0.9, 0], [np.inf, 1.1, np.inf]),
            "shifted_power": ([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
            "vapor_pressure": ([-np.inf] * 3, [np.inf] * 3),
            "heat_capacity": ([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
            "rational": ([-np.inf] * 4, [np.inf] * 4),
            "modified_hoerl": ([0, 0.9, -np.inf], [np.inf, 1.1, np.inf]),
            "sinusoidal": ([-np.inf, 0, 0, -np.pi], [np.inf, np.inf, np.inf, np.pi]),
            "gompertz": ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            "saturation_growth": ([0, 0], [np.inf, np.inf]),
            "gaussian": ([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
            "logistic": ([0, 0, 0], [np.inf, np.inf, np.inf]),
            "modified_geometric": ([0, -np.inf], [np.inf, np.inf]),
            "mmf": ([0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
            "modified_power": ([0, 0.9], [np.inf, 1.1]),
            "modified_exponential": ([0, -np.inf], [np.inf, np.inf])
        }

        self.p0_default = {
            "linear": [1, 1],
            "quadratic": [1, 1, 0.1],
            "cubic": [1, 1, 0.1, 0.01],
            "power": [1, 0.5],
            "geometric": [1, 0.1],
            "logarithmic": [1, 1],
            "exponential": [1, 0.1],
            "bleasdale": [1, 1, 1],
            "richards": [1000, 1, 0.1, 1],
            "weibull": [1000, 1, 0.1, 1],
            "hoerl": [1, 1.0006, 1],
            "shifted_power": [1, 0, 1],
            "vapor_pressure": [1, -100, 1],
            "heat_capacity": [1, 0.1, 1000],
            "rational": [1, 1, 0.1, 0.01],
            "modified_hoerl": [1, 1.0006, 1],
            "sinusoidal": [0, 1, 0.1, 0],
            "gompertz": [1, 1, 0.1],
            "saturation_growth": [1, 1],
            "gaussian": [1, 5, 1],
            "logistic": [1, 1, 0.1],
            "modified_geometric": [1, 1],
            "mmf": [1, 1, 1, 1],
            "modified_power": [1, 1.0006],
            "modified_exponential": [1, 1]
        }

    def _calculate_aicc_p(self, y_true, y_pred, n_params):
        n = len(y_true)  # Sample size
        k = n_params  # Number of parameters

        if n <= k + 1:  # Prevent division by zero
            return np.inf

        rss = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares

        # AICc.p formula
        term1 = n * np.log(rss / n)
        term2 = 2 * k
        term3 = (2 * k * (k + 1)) / (n - k - 1)
        aicc_p = term1 + term2 + term3

        return aicc_p

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.results_ = {}
        fitted_models = []

        for name, func in tqdm(self.models.items(), desc="Fitting models", disable=not self.progress_bar):
            try:
                p0 = self.p0_default[name]
                bounds = self.bounds[name]

                if name in ['richards', 'weibull', 'gompertz', 'logistic', 'saturation_growth']:
                    p0[0] = max(y) * 0.8  # Adjust initial value

                params, _ = curve_fit(func, X.flatten(), y, p0=p0, bounds=bounds, maxfev=10000)
                y_pred = func(X.flatten(), *params)

                if np.any(np.isnan(y_pred)) or np.any(y_pred < 0):
                    continue

                residuals = y - y_pred
                std_dev = np.std(residuals)
                aicc_p = self._calculate_aicc_p(y, y_pred, len(params))

                fitted_models.append({
                    'name': name,
                    'params': params,
                    'func': func,
                    'std_dev': std_dev,
                    'aicc_p': aicc_p
                })

            except Exception as e:
                continue

        # Selection logic: top 3 by std dev, then choose best by AICc.p
        if fitted_models:
            fitted_models.sort(key=lambda x: x['std_dev'])
            top3_std = fitted_models[:3]
            best_model = min(top3_std, key=lambda x: x['aicc_p'])

            self.results_[best_model['name']] = {
                'params': best_model['params'],
                'aicc_p': best_model['aicc_p'],
                'func': best_model['func']
            }

        return self

    def predict(self, X, model_name=None):
        X = check_array(X)
        if not hasattr(self, 'results_'):
            raise NotFittedError("Model not fitted yet")
        if model_name is None:
            model_name = next(iter(self.results_.keys()))
        model = self.results_[model_name]
        return model['func'](X.flatten(), *model['params'])

    def predict_at(self, x_value, model_name=None):
        return float(self.predict(np.array([[x_value]]), model_name))

    def get_best_model_info(self):
        if not hasattr(self, 'results_') or not self.results_:
            return {'model_name': None, 'params': None, 'aicc_p': np.inf}
        best = min(self.results_.items(), key=lambda x: x[1]['aicc_p'])
        return {'model_name': best[0], 'params': best[1]['params'], 'aicc_p': best[1]['aicc_p']}


def load_data(plant1_path, plant2_path, insect_path):
    """Load data ensuring sample IDs are identical across all three files"""
    plant1 = pd.read_csv(plant1_path, index_col=0)
    plant2 = pd.read_csv(plant2_path, index_col=0)
    insect = pd.read_csv(insect_path, index_col=0)

    # Verify sample IDs are consistent
    assert all(plant1.index == plant2.index) and all(plant1.index == insect.index), \
        "Sample IDs are inconsistent across the three data tables!"

    print("\nData check:")
    print(f"Plant data 1 - Samples: {len(plant1)}, Species: {plant1.shape[1]}")
    print(f"Plant data 2 - Samples: {len(plant2)}, Species: {plant2.shape[1]}")
    print(f"Insect data - Samples: {len(insect)}, Species: {insect.shape[1]}")

    return (plant1.values, plant2.values, insect.values,
            plant1.index.tolist(), plant1.shape[1], plant2.shape[1], insect.shape[1])


def analyze_samples(plant1_data, plant2_data, insect_data, sample_names,
                    plant1_target, plant2_target, true_insect_richness,
                    min_samples=12, max_samples=12, reps=100, order_reps=100):
    results = []

    for sample_size in tqdm(range(min_samples, max_samples + 1), desc="Processing samples"):
        # Store all valid curves for subsequent averaging
        all_plant1_curves = []
        all_plant2_curves = []
        all_insect_curves = []

        # Perform independent sampling reps times
        for _ in range(reps):
            # Randomly select samples (ensure plant1 and plant2 use same samples)
            idx = np.random.choice(len(sample_names), sample_size, replace=False)

            # Get data for current sampling
            sampled_plant1 = plant1_data[idx]
            sampled_plant2 = plant2_data[idx]
            sampled_insect = insect_data[idx]

            # Store ordered curves
            plant1_order_curves = []
            plant2_order_curves = []
            insect_order_curves = []

            # Perform order_reps times of ordering
            for _ in range(order_reps):
                # Randomly permute sample order
                order = np.random.permutation(sample_size)

                # Calculate accumulation curves for current order
                # Note: Using insects as X, plants as Y
                insect_curve = np.array([len(np.unique(np.nonzero(sampled_insect[order[:i + 1]])[1]))
                                         for i in range(sample_size)])
                plant1_curve = np.array([len(np.unique(np.nonzero(sampled_plant1[order[:i + 1]])[1]))
                                         for i in range(sample_size)])
                plant2_curve = np.array([len(np.unique(np.nonzero(sampled_plant2[order[:i + 1]])[1]))
                                         for i in range(sample_size)])

                if (np.max(insect_curve) >= 5 and np.max(plant1_curve) >= 5
                        and np.max(plant2_curve) >= 5):
                    insect_order_curves.append(insect_curve)
                plant1_order_curves.append(plant1_curve)
                plant2_order_curves.append(plant2_curve)

            # If valid data exists, calculate average curve for current sampling
            if insect_order_curves:
                all_insect_curves.append(np.mean(insect_order_curves, axis=0))
            all_plant1_curves.append(np.mean(plant1_order_curves, axis=0))
            all_plant2_curves.append(np.mean(plant2_order_curves, axis=0))

        # Check if valid data exists
        if not all_insect_curves:
            results.append({
                'sample_size': sample_size,
                'n_valid': 0,
                'plant1_mean_pred': np.nan,
                'plant2_mean_pred': np.nan,
                'plant1_std_pred': np.nan,
                'plant2_std_pred': np.nan,
                'plant1_dominant_model': None,
                'plant2_dominant_model': None,
                'insect_target_x': true_insect_richness,
                'true_plant1_richness': plant1_target,
                'true_plant2_richness': plant2_target
            })
            continue

        # Calculate average curves (mean of all valid curves)
        avg_insect = np.mean(all_insect_curves, axis=0)
        avg_plant1 = np.mean(all_plant1_curves, axis=0)
        avg_plant2 = np.mean(all_plant2_curves, axis=0)

        # Perform single fitting using average curves
        plant1_pred = np.nan
        plant1_model = None
        plant2_pred = np.nan
        plant2_model = None

        try:
            # Fit plant1 data (insects as X, plant1 as Y)
            ce = CurveExpertEmulator()
            ce.fit(avg_insect.reshape(-1, 1), avg_plant1)
            plant1_pred = ce.predict_at(true_insect_richness)
            plant1_model = ce.get_best_model_info()['model_name']
        except Exception as e:
            print(f"Plant1 fitting error: {e}")

        try:
            # Fit plant2 data (insects as X, plant2 as Y)
            ce = CurveExpertEmulator()
            ce.fit(avg_insect.reshape(-1, 1), avg_plant2)
            plant2_pred = ce.predict_at(true_insect_richness)
            plant2_model = ce.get_best_model_info()['model_name']
        except Exception as e:
            print(f"Plant2 fitting error: {e}")

        results.append({
            'sample_size': sample_size,
            'n_valid': len(all_insect_curves),
            'plant1_mean_pred': plant1_pred,
            'plant2_mean_pred': plant2_pred,
            'plant1_std_pred': np.nan,
            'plant2_std_pred': np.nan,
            'plant1_dominant_model': plant1_model,
            'plant2_dominant_model': plant2_model,
            'insect_target_x': true_insect_richness,
            'true_plant1_richness': plant1_target,
            'true_plant2_richness': plant2_target
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Configure paths
    input_dir = "E:/Curve"
    output_dir = "E:/Curve/May/all/"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    plant1, plant2, insect, samples, plant1_target, plant2_target, true_insect_richness = load_data(
        f"{input_dir}/new_plant2.csv", # Surrounding plant OTU table (Observe/absence)
        f"{input_dir}/May/all/new_May-feed.csv", # Feeding plant OTU table (Observe/absence)
        f"{input_dir}/May/all/May-ins.csv" # Insecta OTU table (Observe/absence)
    )

    # Run analysis
    results = analyze_samples(
        plant1, plant2, insect, samples,
        plant1_target, plant2_target, true_insect_richness,
        min_samples=3, max_samples=24, reps=1000, order_reps=1000
    )

    # Save results
    results.to_csv(f"{output_dir}/May_rev_results.csv", index=False)
    print("\nResults saved to May_rev_results.csv")

    # Print summary
    print("\nResults summary:")
    print(results[['sample_size', 'plant1_mean_pred', 'plant2_mean_pred']])

    # Print model distribution
    print("\nModel distribution:")
    print("Plant1 models:")
    print(results['plant1_dominant_model'].value_counts())
    print("\nPlant2 models:")
    print(results['plant2_dominant_model'].value_counts())


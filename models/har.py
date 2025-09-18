import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class HARModel:
    """
    HAR model using scikit-learn LinearRegression.
    Fits: log(y_t) ~ const + log(RV_d) + log(RV_w) + log(RV_m)
    where RV_d, RV_w, RV_m are the last daily/weekly/monthly volatilities in each sequence.
    """

    def __init__(self, use_log=True, fit_intercept=True):
        self.use_log = use_log
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.fitted = False

    def fit(self, X, y):
        """
        Fit HAR model.
        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Shape [N, seq_len, n_features], must include daily/weekly/monthly volatility features.
        y : np.ndarray | torch.Tensor
            Shape [N] or [N, 1], target next-day volatility.
        """
        if self.use_log:
            X = np.log(np.clip(X, 1e-12, None))
            y = np.log(np.clip(y, 1e-12, None))
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        """
        Predict on new data.
        X: np.ndarray or torch.Tensor or VolatilityDataset
        Returns np.ndarray of predicted RV (not log).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        if self.use_log:
            X = np.log(np.clip(X, 1e-12, None))

        y_hat = self.model.predict(X)
        if self.use_log:
            y_hat = np.exp(y_hat)
        return y_hat

    def coefficients(self):
        """
        Return model parameters as a dictionary.
        """
        if not self.fitted:
            return None
        coef = {
            "beta_d": self.model.coef_[0],
            "beta_w": self.model.coef_[1] if len(self.model.coef_) > 1 else None,
            "beta_m": self.model.coef_[2] if len(self.model.coef_) > 2 else None,
        }
        coef["const"] = self.model.intercept_
        return coef

    def save_model(self, path):
        """
        Save the fitted model to disk using joblib.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before saving the model.")
        model_data = {
            "model": self.model,
            "use_log": self.use_log,
        }
        joblib.dump(model_data, path)

    def evaluate(self, X, y):
        """
        Evaluate model performance using Mean Squared Error (MSE).
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True target values.
        Returns
        -------
        mse : float
            Mean Squared Error of the predictions.
        """

        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse

    @classmethod
    def load_model(cls, path):
        """
        Load a fitted model from disk using joblib.
        """
        model_data = joblib.load(path)
        har_model = cls(use_log=model_data["use_log"])
        har_model.fitted = True
        har_model.model = model_data["model"]

        return har_model

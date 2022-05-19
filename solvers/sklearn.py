from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import Ridge


class Solver(BaseSolver):
    name = "sklearn"
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    parameters = {
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "saga"],
    }

    def __init__(self, solver="svd"):
        self.solver = solver

    def set_objective(self, X, y, reg=1, fit_intercept=False):
        self.X, self.y, self.fit_intercept = X, y, fit_intercept
        self.clf = Ridge(
            fit_intercept=fit_intercept, alpha=reg, solver=self.solver)

    def run(self, n_iter):
        self.clf.fit(self.X, self.y)

    def get_result(self):
        if self.fit_intercept:
            return np.hstack([self.clf.coef_, self.clf.intercept_])
        else:
            return self.clf.coef_.flatten()

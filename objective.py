from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Ridge Regression"

    parameters = {
        "fit_intercept": [False, True],
        "reg": [1, 10]},

    def __init__(self, fit_intercept=False, reg=1):
        self.fit_intercept = fit_intercept
        self.reg = reg

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, theta):
        c = 0
        if self.fit_intercept:
            theta, c = theta[:-1], theta[-1]
        res = self.y - self.X @ theta - c

        return .5 * res @ res + 0.5 * self.reg * theta @ theta

    def to_dict(self):
        return dict(X=self.X, y=self.y,
                    fit_intercept=self.fit_intercept, reg=self.reg)

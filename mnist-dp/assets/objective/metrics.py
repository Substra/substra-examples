from sklearn.metrics import accuracy_score

import substratools as tools


class MnistMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        """Returns the macro-average recall

        :param y_true: actual values from test data
        :type y_true: array_like
        :param y_pred: predicted values from test data
        :type y_pred: array_like
        :rtype: float
        """
        return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    tools.metrics.execute(MnistMetrics())

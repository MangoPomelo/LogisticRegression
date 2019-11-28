import math
import pickle
import numpy as np


sigmoid = lambda z: 1.0 / (1.0 + math.exp(-z))


class BinomialLogisticRegression(object):
    """BinomialLogisticRegression"""
    def __init__(self, lr=0.03, max_epoch=100, tol=0.0001, verbose=False):
        """
        BinomialLogisticRegression

        Args:
            lr: Learning rate.
            max_epoch: Maximum number of iterations for training.
            tol: Tolerance for stopping criteria.
            verbose: Verbosity.
        """
        self.lr = lr

        self.max_epoch = max_epoch
        self.tol = tol
        
        self.verbose = verbose

    def fit(self, x, y):
        n_samples = len(x)
        n_classes = len(x[0]) + 1 

        x = np.column_stack((x, np.ones(n_samples))) # expand 1 column for intercept

        self.coef_ = np.random.rand(n_classes,)

        # LaTex: J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}(y_i\log(h_{\theta}(x_i))+(1-y_i)\log(1-h_{\theta}(x_i))
        cost_func = lambda pred, label: -label * math.log(pred) - (1 - label) * math.log(1 - pred)

        # Define the deivative cost function
        # Latex: \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x_i)-y_i)x_i_j
        derivative_cf = lambda x, y: np.asarray([(
                np.sum(
                        (sigmoid(self.coef_.dot(x[i])) - y[i]) * x[i][j] 
                        for i in range(n_samples)
                    ) / n_samples
            ) for j in range(n_classes)])

        verbose_interval = int(self.max_epoch * 0.05)
        if verbose_interval == 0: verbose_interval = 1

        # Start finding the optimal coefficient and intercept
        epoch = 1
        while True:
            # Jump out if max epoch reached
            if epoch > self.max_epoch: break

            # Update coefficent and intercept
            # LaTex: \theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
            self.coef_ -= self.lr * derivative_cf(x, y)

            # Calculate average cost for an epoch
            average_epoch_cost = np.average([
                    cost_func(sigmoid(self.coef_.dot(x[i])), y[i])
                    for i in range(n_samples)
                ])

            # For verbose
            if self.verbose and not epoch % verbose_interval:
                print(f"Epoch: {epoch:<10} | Loss : {average_epoch_cost:<6.4f}")

            # Jump out if cost less than tolerance
            if average_epoch_cost <= self.tol: break 
            epoch += 1
        
    def predict_proba(self, x):
        n_samples = len(x)
        x = np.column_stack((x, np.ones(n_samples)))

        return np.asarray([sigmoid(self.coef_.dot(x[i])) for i in range(n_samples)])

    def predict(self, x):
        return np.asarray([1 if proba >= 0.5 else 0 for proba in self.predict_proba(x)])

    def score(self, x, y):
        predictions = self.predict(x)
        return np.average(np.logical_not(np.logical_xor(predictions, y)))

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        return self

    @classmethod
    def load(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42) # 42, the solution to the whole universe

    from sklearn.model_selection import train_test_split

    x = np.random.randint(low=0, high=4, size=(200,4)) # shape = (200, 4)
    y = np.array([1 if sum(features) >= 6 else 0 for features in x]).reshape(-1,) # shape = (200,)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = BinomialLogisticRegression(max_epoch=500, verbose=True)
    clf.fit(x_train, y_train)

    # Save'n'load
    clf.save("./blr.pkl")
    clf = BinomialLogisticRegression.load("./blr.pkl")

    pred = clf.predict(x_test)
    acc = clf.score(x_test, y_test)

    print(f"Score:{acc:.2f}")
    print("Predictions:", pred[:5])
    print("True values:", y_test[:5])

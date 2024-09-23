import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        return np.mean(iris[0:4], axis=0)

    def empirical_covariance(self, iris):
        return np.cov(iris[:, :4], rowvar=False)

    def feature_means_class_1(self, iris):
        return np.mean(iris[iris[:, 4] == 1][:, :4], axis=0)

    def empirical_covariance_class_1(self, iris):
        return np.cov(iris[iris[:, 4] == 1][:, :4], rowvar=False)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))

    def manhattan_dist(self, p, x):
        return np.sum(np.abs(x - p), axis=1)

    def predict(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
           distances = self.manhattan_dist(ex, self.train_inputs)
           M = len(distances)

           ind_neighbors = np.array([j for j in range(M) if distances[j] < self.h])

           if len(ind_neighbors) == 0:
                classes_pred[i] = draw_rand_label(ex, self.label_list)
                continue

           for k in ind_neighbors:
                counts[i, int(self.train_labels[k]) - 1] += 1

           classes_pred[i] = np.argmax(counts[i, :]) + 1

        return classes_pred






class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def predict(self, test_data):
        pass


def split_dataset(iris):
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        pass

    def soft_parzen(self, sigma):
        pass


def get_test_errors(iris):
    pass


def random_projections(X, A):
    pass

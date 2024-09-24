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

def dist_func(p, x):
    return np.sum(np.abs(x - p), axis=1)

def kernel_func(d, sigma, distance):
    ((2*np.pi) ** d/2) * (sigma ** d) / np.exp(-0.5 * ((distance**2)/(sigma**2)))



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


    def predict(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
           distances = dist_func(ex, self.train_inputs)
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
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))

    def predict(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
           distances = dist_func(ex, self.train_inputs)
           M = len(distances)

           ind_neighbors = np.array([j for j in range(M) if distances[j] < self.sigma])

           for k in ind_neighbors:
                counts[i, int(self.train_labels[k]) - 1] += kernel_func(len(self.train_inputs[0]), self.sigma, test_data.shape[1]  )

           classes_pred[i] = np.argmax(counts[i, :]) + 1

        return classes_pred


def split_dataset(iris):

    train_set = np.empty(shape=(0,5))
    validation_set = np.empty(shape=(0,5))
    test_set = np.empty(shape=(0,5))

    train_inds = []
    validation_inds =[]
    test_inds = []

    for i in range(len(iris)):
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            train_inds.append(i)

        if i % 5 == 3:
            validation_inds.append(i)

        if i % 5 == 4:
            test_inds.append(i)

    train_set = np.array([iris[i] for i in train_inds])
    validation_set = np.array([iris[i] for i in validation_inds])
    test_set = np.array([iris[i] for i in test_inds])


    return train_set, validation_set, test_set



class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        classifier = HardParzen(h)
        classifier.fit(self.x_train, self.y_train)

        classes_pred = classifier.predict(self.x_val)

        count = 0

        for i in range(len(classes_pred)):
            if classes_pred[i] != self.y_val[i]:
                count += 1

        return count / len(classes_pred)

    def soft_parzen(self, sigma):
        classifier = SoftRBFParzen(sigma)
        classifier.fit(self.x_train, self.y_train)

        classes_pred = classifier.predict(self.x_val)

        count = 0

        for i in range(len(classes_pred)):
            if classes_pred[i] != self.y_val[i]:
                count += 1

        return count / len(classes_pred)


def get_test_errors(iris):
    train_set, validation_set, test_set = split_dataset(iris)

    x_train = train_set[:, :4]
    y_train = train_set[:, 4]

    x_val = validation_set[:, :4]
    y_val = validation_set[:, 4]

    err_rate = ErrorRate(x_train, y_train, x_val, y_val)

    h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    temp = np.inf
    min_h = np.inf
    min_sigma = np.inf

    for h in h_values:
        error = err_rate.hard_parzen(h)
        if error < temp:
            temp = error
            min_h = h

    temp = np.inf

    for sigma in sigma_values:
        error = err_rate.soft_parzen(sigma)
        if error < temp:
            temp = error
            min_sigma = sigma

    return min_h, min_sigma




def random_projections(X, A):
    pass


if __name__ == '__main__':
    data = np.genfromtxt('iris.txt')


    print(get_test_errors(data))
import sklearn
import pandas as pd


class BaseLearner(object):
    def __init__(self, learner, train_pred_path, test_pred_path):
        self.learner = learner
        self.train_pred_path = train_pred_path
        self.test_pred_path = test_pred_path


class StackLearner(object):
    def __init__(self, stack_learner, base_learners=None, train_set, test_set, target):
        self.base_learners = base_learners
        self.n_folds = len(self.base_learners)
        self.target = target
        self.x_original_train = train_set[:]
        self.y_original_train = train_set[target].values
        self.x_original_test = test_set[:]

    @classmethod
    def stacking_partition(self, x, y, n_folds, index):
        pass

    def fit_base_learners(self):
        pred_original_test = np.zeros(len(self.x_original_test))
        self.stack_train = pd.DataFrame()
        for (index, learner) in enumerate(self.base_learners):
            x_train, y_train, x_test, y_test = StackLearner.stacking_partition(self.x_original_train, self.y_original_train, self.n_folds, index)
            learner.fit(x_train, y_train)
            # change here!
            self.stack_train.concat(pd.DataFrame({'ID': x_test.ID.values, self.target: learner.pred(x_test)}))
            self.pred_original_test += learner.pred(x_original_test)
        self.stack_test = pred_original_test /= float(self.n_folds)

    def fit_stack_learner(self):
        self.stack_learner.fit(self.stack_train, self.)

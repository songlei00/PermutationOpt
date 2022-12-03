import numpy as np
import logging

log = logging.getLogger(__name__)


def cal_idx(i, j, dims):
    i, j = min(i, j), max(i, j)
    return int( (2*dims-i-1)*i/2 + j-i-1 )


class OrderConstraint:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def check(self, x, feature_x):
        if x[self.first] > x[self.second]:
            return True
        else:
            return False


class PositionConstraint:
    def __init__(self, idx, feasible_pos):
        self.idx = idx
        self.feasible_pos = feasible_pos

    def check(self, x, feature_x):
        if x[self.idx] in self.feasible_pos:
            return True
        else:
            return False


class Node:
    def __init__(self, dims, parent, leaf_size):
        self.dims = dims
        self.parent = parent
        self.leaf_size = leaf_size
        self.depth = 0 if parent is None else parent.depth + 1
        # v and n for UCB
        self.v = None
        self.n = None
        self.is_splittable = False

        self.constraints = {'order': [], 'position': []}
        
        self.kids = [] # 0: good, 1: bad

        self.train_X = []
        self.feature_X = []
        self.train_Y = []

    def get_uct(self, Cp):
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        return self.v + 2*Cp*np.sqrt( 2* np.power(self.parent.n, 0.5) / self.n )

    def update_bag(self, train_X, feature_X, train_Y):
        self.train_X.extend(train_X)
        self.feature_X.extend(feature_X)
        self.train_Y.extend(train_Y)

        self.v = np.mean(train_Y)
        self.n = 0
        if len(self.train_X) < self.leaf_size:
            self.is_splittable = False
        else:
            self.is_splittable = True

    def add_constraint(self, constraint_type, constraint):
        self.constraints[constraint_type].append(constraint)

    def divide_data(self, train_X, feature_X, train_Y):
        mean_y = np.mean(train_Y)

        good_idx = [idx for idx, y in enumerate(train_Y) if y >= mean_y]
        good_train_X, good_feature_X, good_train_Y = [], [], []
        for idx in good_idx:
            good_train_X.append(train_X[idx])
            good_feature_X.append(feature_X[idx])
            good_train_Y.append(train_Y[idx])

        bad_idx = [idx for idx, y in enumerate(train_Y) if y < mean_y]
        bad_train_X, bad_feature_X, bad_train_Y = [], [], []
        for idx in bad_idx:
            bad_train_X.append(train_X[idx])
            bad_feature_X.append(feature_X[idx])
            bad_train_Y.append(train_Y[idx])
        
        return good_train_X, good_feature_X, good_train_Y, \
            bad_train_X, bad_feature_X, bad_train_Y

    def find_common_pattern(self, pattern_type, train_X, feature_X, good_train_X, good_feature_X, n=1):
        # 现在只能找到good里面的特征，但可能bad也满足，所以
        # 感觉还是需要找good和bad存在的差别，作为constraint
        # 先找到差别，然后看good中的顺序是什么样的

        # find the common pattern in the data, and use the pattern as a 
        # constraint of the good data
        assert pattern_type in ['order', 'position']
        if pattern_type == 'order':
            score = np.abs(np.sum(feature_X, axis=0))
            max_idx = np.argmin(score)
            # print(score[max_idx])
            for i in range(self.dims):
                cnt = (2*self.dims-i-1)*i/2
                if cnt > max_idx:
                    break
            i = i-1
            cnt = (2*self.dims-i-1)*i/2
            j = int(max_idx - cnt + i + 1)
            log.info('pattern: {}, {}, {}'.format(max_idx, i, j))
            assert cal_idx(i, j, self.dims) == max_idx
            constraint = OrderConstraint(i, j)

            # 
            total = len(good_train_X)
            feasible = 0
            for x in good_train_X:
                feasible += constraint.check(x, None)
            if feasible / total < 0.5:
                constraint = OrderConstraint(j, i)
        elif pattern_type == 'position':
            constraint = PositionConstraint()
        else:
            raise NotImplementedError
        return constraint

    def split(self):
        good_train_X, good_feature_X, good_train_Y, bad_train_X, bad_feature_X, bad_train_Y \
            = self.divide_data(self.train_X, self.feature_X, self.train_Y)
        pattern_type = 'order'
        # constraint = self.find_common_pattern(pattern_type, good_train_X, good_feature_X)
        constraint = self.find_common_pattern(pattern_type, self.train_X, self.feature_X, good_train_X, good_feature_X)
        if pattern_type == 'order':
            inv_constraint = OrderConstraint(constraint.second, constraint.first)
        elif pattern_type == 'position':
            inv_constraint = PositionConstraint(constraint.idx, set(range(self.dims)) - constraint.feasible_pos)
        else:
            raise NotImplementedError
        
        good_kid = Node(self.dims, self, self.leaf_size)
        bad_kid = Node(self.dims, self, self.leaf_size)

        # add constraints to the new node
        for cons_type in self.constraints.keys():
            for cons in self.constraints[cons_type]:
                good_kid.add_constraint(cons_type, cons)
                bad_kid.add_constraint(cons_type, cons)
        good_kid.add_constraint(pattern_type, constraint)
        bad_kid.add_constraint(pattern_type, inv_constraint)

        good_train_X, good_feature_X, good_train_Y = [], [], []
        bad_train_X, bad_feature_X, bad_train_Y = [], [], []
        for x, feature_x, y in zip(self.train_X, self.feature_X, self.train_Y):
            is_feasible = constraint.check(x, feature_x)
            if is_feasible:
                good_train_X.append(x)
                good_feature_X.append(feature_x)
                good_train_Y.append(y)
            else:
                bad_train_X.append(x)
                bad_feature_X.append(feature_x)
                bad_train_Y.append(y)
        print('len good train X: {}'.format(len(good_train_X)))
        assert len(good_train_X) != 0 and len(good_train_X) != len(self.train_X)
        good_kid.update_bag(good_train_X, good_feature_X, good_train_Y)
        bad_kid.update_bag(bad_train_X, bad_feature_X, bad_train_Y)
        self.is_splittable = False
        self.kids = [good_kid, bad_kid]
        
        return good_kid, bad_kid




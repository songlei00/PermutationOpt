import numpy as np
from typing import List
import time
import logging
from ._base import BaseOptimizer
from ._node import Node
from ._ea_operator import swap_mutation
from ._utils import get_init_samples, featurize

log = logging.getLogger(__name__)


class MCTS(BaseOptimizer):
    def __init__(self, dims, lb, ub, Cp=1, leaf_size=20, max_propose=200):
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.Cp = Cp
        self.leaf_size = leaf_size
        self.max_propose = max_propose

        self.solver_type = 'sa'
        self.solver_config = {
            'T': 100,
            'decay': 0.99,
            'update_freq': 100,
        }
        self.reset_solver()

        self.train_X = []
        self.feature_X = []
        self.train_Y = []
        self.best_x = None
        self.best_y = None
        self.n_propose = 0

        self.nodes = []
        root = Node(self.dims, None, self.leaf_size)
        self.nodes.append(root)
        self.root = root

    def populate_training_data(self):
        self.nodes.clear()
        root = Node(self.dims, None, self.leaf_size)
        self.nodes.append(root)
        self.root = root
        self.root.update_bag(self.train_X, self.feature_X, self.train_Y)

    def get_split_idx(self):
        split_idx = []
        for idx, node in enumerate(self.nodes):
            if node.is_splittable:
                split_idx.append(idx)
        return split_idx

    def is_splittable(self):
        split_idx = self.get_split_idx()
        if len(split_idx) > 0:
            return True
        else:
            return False

    def dynamic_treeify(self):
        self.populate_training_data()
        
        while self.is_splittable():
            to_split_idx = self.get_split_idx()
            print(to_split_idx)
            # if to_split_idx[0] > 1:
            #     assert 0
            for idx in to_split_idx:
                parent = self.nodes[idx]
                good_kid, bad_kid = parent.split()
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

    def select(self):
        curr_node = self.root
        path = []
        while len(curr_node.kids) > 0:
            uct = [node.get_uct(self.Cp) for node in curr_node.kids]
            choice = np.random.choice(np.argwhere(uct == np.amax(uct)).reshape(-1), 1)[0]
            path.append( (curr_node, choice) )
            curr_node = curr_node.kids[choice]
        return curr_node, path

    def check_constraints(self, x, feature_x, constraints):
        for cons in constraints:
            if not cons.check(x, feature_x):
                return False
        return True

    def optimize(self, node):
        if len(self.train_X) == 0:
            next_x = get_init_samples('permutation', 1, self.dims, self.lb, self.ub)[0]
            next_feature_x = None
            return next_x, next_feature_x

        best_idx = np.argmax(node.train_Y)
        best_x = node.train_X[best_idx]
        constraints = node.constraints
        order_constraints = constraints['order']
        position_constraints = constraints['position']

        if self.solver_type == 'sa':
            while True:
                next_x = swap_mutation(best_x)
                # next_feature_x = featurize(next_x, 'numpy')
                is_order_feasible = self.check_constraints(next_x, None, order_constraints)
                is_pos_feasible = self.check_constraints(next_x, None, position_constraints)
                if is_order_feasible and is_pos_feasible:
                    break
            # 没有做sa中的按概率接受差的解
            self.solver_state['step'] += 1
            if self.solver_state['step'] % self.solver_config['update_freq'] == 0:
                self.solver_state['T'] *= self.solver_config['decay']
        else:
            raise NotImplementedError

        return next_x, None

    def backpropogate(self, leaf, X, feature_X, Y):
        curr_node = leaf
        while curr_node is not None:
            curr_node.v = (curr_node.v * curr_node.n + np.sum(Y)) / (curr_node.n + len(Y))
            curr_node.n += len(Y)
            curr_node.update_bag(X, feature_X, Y)
            curr_node = curr_node.parent

    def ask(self) -> List[np.ndarray]:
        if self.n_propose == 0:
            self.dynamic_treeify()
            self.leaf, path = self.select()
            self.reset_solver()
        next_x, next_feature_x = self.optimize(self.leaf)
        return [next_x]

    def tell(self, X: List[np.ndarray], Y):
        self.n_propose = (self.n_propose + 1) % self.max_propose
        self.train_X.extend(X)
        feature_X = [featurize(x, 'numpy') for x in X]
        self.feature_X.extend(feature_X)
        self.train_Y.extend(Y)

        self.backpropogate(self.leaf, X, feature_X, Y)

    def reset_solver(self):
        self.solver_state = {
            'T': self.solver_config['T'],
            'step': 0,
        }
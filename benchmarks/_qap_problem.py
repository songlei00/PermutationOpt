import numpy as np
from scipy import io


class QAPProblem:
    def __init__(self, benchmark_index):
        self.A = np.asarray(io.loadmat('./data/qap/QAP_LIB_A'+str(benchmark_index+1)+'.mat')['A'])
        self.B = np.asarray(io.loadmat('./data/qap/QAP_LIB_'+str(benchmark_index+1)+'.mat')['B'])
        self.dims = self.A.shape[0]
        
    def __call__(self, x):
        assert len(set(x)) == self.dims
        assert len(x) == self.dims
        x = x.astype(np.int)
        E = np.eye(self.dims)
        # permutation = np.array([np.arange(self.dims), x]) 
        permutation = x

        P = np.zeros([self.dims, self.dims])
        for i in range(self.dims):
            # P[:, i] = E[:, permutation[1][i]]
            P[:, i] = E[:, x[i]]
        result = (np.trace(P.dot(self.B).dot(P.T).dot(self.A.T)))
        return result
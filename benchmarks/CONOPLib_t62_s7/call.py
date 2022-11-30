import ctypes
import os
from savedat import savedat
import pandas as pd
import numpy as np
import configparser
import logging

log = logging.getLogger(__name__)


class EarthBenchmark:
    def __init__(self):
        self.so = ctypes.CDLL('./CONOPLib.so')
        self.nevent = 124
        self.position = np.zeros(self.nevent)

        # FadLad and coex are used to check the constraints
        df = pd.read_csv('Fb4L.dat', delim_whitespace=True, header=None)
        self.FadLad = df.loc[:].values
        df = pd.read_csv('coex.dat', delim_whitespace=True, header=None)
        self.coex = df.loc[:].values
        for i in range(0, len(self.FadLad)):
            self.FadLad[i][0] = self.FadLad[i][0] * 2
            self.FadLad[i][1] = self.FadLad[i][1] * 2 +1

        for i in range(0, len(self.coex)):
            self.coex[i][0] = self.coex[i][0] * 2
            self.coex[i][1] = self.coex[i][1] * 2

    def __call__(self, x):
        assert x.ndim == 1
        # check and repair
        log.debug('checkFADLAD: {}'.format(self.checkFADLAD(x)))
        log.debug('checkcoex: {}'.format(self.checkcoex(x)))
        self.repair(x)
        assert self.checkFADLAD(x) == True and self.checkcoex(x) == True

        # evaluate
        carrary = (ctypes.c_int * len(x))(*x)
        self.so.getPenalty.restype = ctypes.c_double
        pen1 = self.so.getPenalty(carrary, len(x))
        return pen1

    def loadsoln(self):
        soln = np.zeros(self.nevent)
        df = pd.read_csv('soln.dat', delim_whitespace=True, header=None)
        soln_matrix = df.loc[:].values
        for i in range(0, len(soln_matrix)):
            soln[i] = (soln_matrix[i][0]-1)*2 + soln_matrix[i][1] - 1
        return soln

    def getPosition(self, soln):
        for i in range(0, self.nevent):
            self.position[soln[i]] = i

    def checkFADLAD(self, soln):
        self.getPosition(soln)
        for i in range(0, len(self.FadLad)):
            if self.position[self.FadLad[i][0]] > self.position[self.FadLad[i][1]]:
                return False
        return True

    def checkcoex(self, soln):
        self.getPosition(soln)
        for i in range(0, len(self.coex)):
            if self.position[self.coex[i][0]] > self.position[self.coex[i][1] + 1] or self.position[self.coex[i][1]] > self.position[self.coex[i][0] + 1]:
                return False
        return True

    def repair(self, soln):
        self.getPosition(soln)
        rank = 0
        # solve fadlad for the same taxa
        for i in range(0, self.nevent - 1, 2):
            if self.position[i] > self.position[i + 1]:
                soln[int(self.position[i])], soln[int(self.position[i + 1])] = soln[int(self.position[i + 1])], soln[int(self.position[i])]
        while rank != self.nevent:
            # if perm[rank] is FAD
            if(soln[rank] % 2 == 0):
                # search for LAD
                lrank = int(self.position[soln[rank] + 1])
                trank = lrank + 1
                while(trank != self.nevent):
                    if(soln[trank] % 2 == 0 and ([soln[trank], soln[lrank]] in self.FadLad.tolist() or [soln[trank], soln[rank]] in self.coex.tolist() or [soln[trank], soln[rank]] in coex.tolist())):
                        soln[trank], soln[lrank] = soln[lrank], soln[trank]
                        self.position[soln[lrank]], self.position[soln[trank]] = self.position[soln[trank]], self.position[soln[lrank]]
                        # getPosition(soln)
                    trank += 1
            rank += 1
        

if __name__ == '__main__':
    x1 = [2, 4, 3, 5, 6, 7, 8, 14, 15, 10, 9, 12, 26, 16, 18, 22, 13, 17, 19, 28, 32, 24, 30, 40, 44, 46, 0, 48, 42, 36, 20, 37, 31, 23, 122, 38, 52, 34, 50, 56, 21, 1, 29, 39, 25, 53, 123, 45, 58, 27, 62, 54, 47, 57, 11, 59, 55, 66, 60, 68, 72, 64, 73, 63, 65, 67, 76, 51, 35, 33, 41, 70, 78, 71, 79, 82, 74, 84, 61, 77, 80, 88, 75, 43, 69, 92, 86, 90, 87, 91, 112, 93, 49, 89, 108, 104, 85, 113, 96, 110, 111, 94, 102, 103, 98, 100, 106, 97, 95, 114, 107, 101, 105, 109, 116, 81, 83, 118, 99, 120, 119, 115, 121, 117]
    x2 = [22, 4, 8, 2, 6, 5, 16, 10, 12, 14, 18, 15, 26, 19, 17, 36, 44, 20, 9, 34, 13, 32, 58, 30, 7, 28, 46, 40, 24, 52, 42, 56, 50, 3, 72, 48, 0, 54, 53, 122, 38, 25, 39, 31, 27, 29, 45, 60, 70, 62, 59, 76, 68, 92, 55, 64, 78, 11, 96, 90, 66, 51, 118, 88, 1, 35, 41, 47, 112, 57, 98, 82, 104, 80, 74, 79, 106, 94, 63, 71, 84, 86, 73, 87, 110, 93, 61, 67, 23, 116, 37, 75, 100, 33, 114, 108, 69, 43, 97, 102, 21, 105, 101, 111, 91, 113, 123, 109, 81, 120, 107, 121, 65, 89, 99, 85, 119, 117, 95, 49, 83, 77, 103, 115]

    benchmark = EarthBenchmark()
    y1 = benchmark(x1)
    y2 = benchmark(x2)
    print(y1)
    print(y2)
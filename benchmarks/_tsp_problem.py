import numpy as np
import logging

log = logging.getLogger(__name__)
PI = 3.141592


def nint(x):
    return int(x + 0.5)

def euc_2d_dis_fn(x1, y1, x2, y2):
    xd = x1 - x2
    yd = y1 - y2
    return nint(np.sqrt(xd**2 + yd**2))


def cal_latitude(x):
    deg = nint(x)
    min = x - deg
    lat = PI * (deg + 5.0 * min / 3) / 180
    return lat


def cal_longitude(y):
    deg = nint(y)
    min = y - deg
    lon = PI * (deg + 5.0 * min / 3) / 180
    return lon


def geo_dis_fn(x1, y1, x2, y2):
    latitude1 = cal_latitude(x1)
    longitude1 = cal_longitude(y1)
    latitude2 = cal_latitude(x2)
    longitude2 = cal_longitude(y2)
    RRR = 6378.388
    q1 = np.cos(longitude1 - longitude2)
    q2 = np.cos(latitude1 - latitude2)
    q3 = np.cos(latitude1 + latitude2)
    d = int(RRR * np.arccos(0.5 * ((1+q1)*q2 - (1-q1)*q3)) + 1)
    return d


def att_dis_fn(x1, y1, x2, y2):
    xd = x1 - x2
    yd = y1 - y2
    r = np.sqrt((xd*xd + yd*yd) / 10)
    t = nint(r)
    if t < r:
        d = t + 1
    else:
        d = t
    return d


class TSPProblem:
    def __init__(self, file_path, maximize=True, is_load_opt=False):
        self.meta_info_dict, self.pos_dict = self._read_config_file(file_path + '.tsp')
        assert self.meta_info_dict['TYPE'] == 'TSP'
        log.debug('meta info dict')
        log.debug(self.meta_info_dict)
        log.debug('pos dict')
        log.debug(self.pos_dict)
        self.dims = int(self.meta_info_dict['DIMENSION'])
        self.maximize = maximize

        # calculate the distance
        self.dis_fn = self._get_dis_fn(self.meta_info_dict['EDGE_WEIGHT_TYPE'])
        self.dis_mat = self._get_dis_mat()

        # best solution
        if is_load_opt:
            _, self.opt = self._read_opt_file(file_path + '.opt.tour')
            log.debug('opt: {}'.format(self.opt))
            self.opt_len = self.__call__(self.opt)
            log.debug('opt length: {}'.format(self.opt_len))

    def _read_config_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.readlines()
        meta_info_dict = dict()
        pos_dict = dict()
        for line in content:
            line = line.strip()
            if len(line) == 0 or line in ['NODE_COORD_SECTION', 'EOF']:
                pass
            elif line[0].isdigit():
                # parse the position
                node_idx, x, y = line.split()
                pos_dict[int(node_idx)] = (float(x), float(y))
            else:
                # parse meta information
                line = [i.strip() for i in line.split(':')]
                key, value = line
                meta_info_dict[key] = value

        assert len(pos_dict) == int(meta_info_dict['DIMENSION'])
        return meta_info_dict, pos_dict

    def _get_dis_fn(self, dis_type):
        if dis_type == 'EUC_2D':
            dis_fn = euc_2d_dis_fn
        elif dis_type == 'GEO':
            dis_fn = geo_dis_fn
        elif dis_type == 'ATT':
            dis_fn = att_dis_fn
        else:
            raise NotImplementedError

        return dis_fn

    def _get_dis_mat(self):
        dim = int(self.meta_info_dict['DIMENSION'])
        dis_mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                x1, y1 = self.pos_dict[i+1]
                x2, y2 = self.pos_dict[j+1]
                dis_mat[i][j] = self.dis_fn(x1, y1, x2, y2)
        return dis_mat

    def _read_opt_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.readlines()
        meta_info_dict = dict()
        opt = []
        for line in content:
            line = line.strip()
            if len(line) == 0 or line in ['TOUR_SECTION', 'EOF']:
                pass
            elif line[0].isdigit() or line == '-1':
                # parse the position
                if int(line) == -1:
                    continue
                opt.append(int(line)-1)
            else:
                # parse meta information
                line = [i.strip() for i in line.split(':')]
                key, value = line
                meta_info_dict[key] = value
        assert int(meta_info_dict['DIMENSION']) == len(opt)
        return meta_info_dict, np.array(opt)

    def _evaluate(self, dis_mat, x):
        assert x.ndim == 1
        assert len(x) == int(self.meta_info_dict['DIMENSION'])
        length = 0
        # x = [int(i-1) for i in x]
        x = [int(i) for i in x]
        for i in range(len(x)-1):
            length += dis_mat[x[i]][x[i+1]]
        length += dis_mat[x[-1]][x[0]]
        if self.maximize:
            length = -length
        return length

    def __call__(self, x):
        return self._evaluate(self.dis_mat, x)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base_path = './tsp_data/'
    task_name = ['ulysses22', 'att48', 'lin105', 'lin318', 'rat575', 'pcb1173']
    tsp = TSPProblem(base_path + task_name[5])
    # print(tsp.opt_len)
    print(tsp(np.random.permutation(tsp.dims)))
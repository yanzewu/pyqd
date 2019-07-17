
import sys
import copy
import numpy as np


class Dumper:

    def __init__(self, filename):
        self.filename = filename

    def write_data(self, data, title=None, title_prefix=None):
        
        if title_prefix is None:
            m_title = title
        else:
            m_title = [title_prefix + str(i) for i in range(data.shape[1])]

        fp = open(self.filename) if self.filename != '-' else sys.stdout

        fp.write('\t'.join(m_title) + '\n')
        for row in data:
            fp.write('\t'.join(('%.6g' % d for d in row)))
            fp.write('\n')

        if fp != sys.stdout:
            fp.close()

    def write_data_with_time(self, data, time, title=None, title_prefix=None):
        
        if title_prefix is None:
            m_title = ['t'] + title
        else:
            m_title = ['t'] + [title_prefix + str(i) for i in range(data.shape[1])]

        m_data = np.hstack((time[:,None], data))
        self.write_data(m_data, title=m_title)


def write_scatter_result(output, k, E, stat_matrices):
        
    title = ['k', 'E']
    for i in range((stat_matrices[0].shape[0] - 1)//2):
        for j in range(stat_matrices[0].shape[1]):
            title.append('%dL:%d' % (i+1, j))
        for j in range(stat_matrices[0].shape[1]):
            title.append('%dR:%d' % (i+1, j))

    for j in range(stat_matrices[0].shape[1]):
        title.append('N:%d' % j)

    
    data = np.zeros((len(k), 2 + stat_matrices[0].shape[0]*stat_matrices[0].shape[1]))
    for i, (k_, E_, sm_) in enumerate(zip(k, E, stat_matrices)):
        data[i, 0] = k_
        data[i, 1] = E_
        data[i, 2:] = sm_.flatten()

    dumper = Dumper(output)
    dumper.write_data(data, title=title)


def write_population_result(output, time, pop_matrix):

    dumper = Dumper(output)
    dumper.write_data_with_time(pop_matrix, time, title_prefix='P')
    
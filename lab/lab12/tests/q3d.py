OK_FORMAT=True
test = {   'name': 'q3d',
    'points': None,
    'suites': [   {   'cases': [   {'code': '>>> cnf_matrix_norm.shape == (2, 2)\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.all(np.isclose(cnf_matrix_norm.sum(axis=1), [1, 1]))\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.isclose(cnf_matrix_norm[0, 0], 0.9550561797752809)\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.isclose(cnf_matrix_norm[1, 1], 0.8333333333333334)\nTrue', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

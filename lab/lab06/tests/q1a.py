OK_FORMAT=True
test = {   'name': 'q1a',
    'points': None,
    'suites': [   {   'cases': [   {'code': '>>> X.shape == (392,2)\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> (add_intercept(np.array([[1, 2, 3],[4, 5, 6]]).T)[:,0] == np.ones((3,))).all()\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> add_intercept(np.array([[1, 2, 3],[4, 5, 6]]).T).shape == (3,3)\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> (add_intercept(np.array([[1, 2, 3],[4, 5, 6]]).T)[:,2] == np.array([4,5,6])).all()\nTrue', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

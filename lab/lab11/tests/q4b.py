OK_FORMAT=True
test = {   'name': 'q4b',
    'points': 1,
    'suites': [   {   'cases': [   {'code': '>>> all(np.isclose([sum(u2[:, i] ** 2) for i in range(u2.shape[1])], [1, 1, 1]))\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> all(np.isclose([sum(vt2[i, :] ** 2) for i in range(vt2.shape[1])], [1, 1, 1]))\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> len(s2) == 3\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> (np.isclose(u2 @ np.diag(s2) @ vt2, surfboard_centered)).all()\nTrue', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

OK_FORMAT=True
test = {   'name': 'q1a',
    'points': None,
    'suites': [   {   'cases': [   {'code': '>>> iris_mean.size == 4\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.all(np.isclose(iris_mean, np.array([5.84333333, 3.05733333, 3.758, 1.19933333])))\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.all(np.isclose(iris_std, np.array([0.82530129, 0.43441097, 1.75940407, 0.75969263])))\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.all(np.isclose(np.zeros(4), np.mean(iris_standardized, axis = 0))) # make sure data is centered at 0\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> np.all(np.isclose(np.ones(4), np.std(iris_standardized, axis = 0))) # make sure data has SD 1\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> -2.56 < np.sum(iris_standardized[0]) < -2.5\nTrue', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

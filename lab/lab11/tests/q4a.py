OK_FORMAT=True
test = {   'name': 'q4a',
    'points': 1,
    'suites': [   {   'cases': [   {'code': '>>> len(surfboard_mean) == 3\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> abs(surfboard_mean[0] - 0.0278) < 1e-4\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> abs(surfboard_mean[1] - 0.0332) < 1e-4\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> abs(surfboard_mean[2] - (-0.0203)) < 1e-4\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> isinstance(surfboard_centered, pd.DataFrame)\nTrue', 'hidden': False, 'locked': False},
                                   {'code': '>>> all(np.isclose(np.mean(surfboard_centered), np.array([0, 0, 0])))\nTrue', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

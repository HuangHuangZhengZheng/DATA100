OK_FORMAT=True
test = {   'name': 'q2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> test_model = MyLinearModel()\n>>> test_model.fit(l1, one_hot_X, tips)\n>>> len(test_model.predict(one_hot_X)) == 244\nTrue',
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> test_model = MyLinearModel()\n>>> test_model.fit(l2, one_hot_X, tips)\n>>> len(test_model.predict(one_hot_X)) == 244\nTrue',
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

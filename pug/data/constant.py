import os

MODULE_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(MODULE_PATH, 'cache')
DICT_ROMAN2INT = {'I': 1, 'II': 2, 'III': 3, 'IV': 4,  'V': 5,
                  'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
for num_X in range(1, 5):
    for s, num in DICT_ROMAN2INT.items():
        DICT_ROMAN2INT['X' * num_X + s] = 10 + num

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from inventory import inventory_policy


def test_policy_is_positive():
    policy = inventory_policy([20, 25, 22, 30, 27, 24, 26])
    assert policy['reorder_point'] > 0
    assert policy['target_stock'] > policy['reorder_point']

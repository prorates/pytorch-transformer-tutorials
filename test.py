#!/usr/bin/env python3

from config import get_config
from config import get_device

from tutorial1 import test_model1
from tutorial2 import test_model2
from tutorial3 import test_model3
from tutorial4 import test_model4
from tutorial5 import test_model5
from tutorial6 import test_model6
from tutorial7 import test_model7

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    device = get_device()

    test_model1(config, device)
    test_model2(config, device)
    test_model3(config, device)
    test_model4(config, device)
    test_model5(config, device)
    test_model6(config, device)
    test_model7(config, device)

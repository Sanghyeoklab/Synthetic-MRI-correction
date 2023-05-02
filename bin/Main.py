from Train import Train 
from Test import Test
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
import Util
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Synthetic MRI artifact correction', description = 'Training and test')
    parser.add_argument('--config', help = 'Get parameters', default = 'Config/config.yaml')
    args = parser.parse_args()
    yaml_file = args.config
    args = Util.make_args(Util.yaml2dic(args.config))
    shutil.copyfile(yaml_file, args.Save + "config.yaml")
    Train(args)
    Test(args)










































































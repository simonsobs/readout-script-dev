#ocs_test.py
import argparse
from sodetlib.det_config  import DetConfig

parser = argparse.ArgumentParser()

parser.add_argument('--slot', type=int)
parser.add_argument('--bgs', nargs='+', default=None, type=int)
parser.add_argument('--temp', type=int)
args = parser.parse_args()

cfg = DetConfig()
cfg.load_config_files(slot=args.slot)

if args.bgs is None:
    bgs = range(12)
else:
    bgs = args.bgs

for bg in bgs:
    print(bg)
print(args.temp)
print('Success!')

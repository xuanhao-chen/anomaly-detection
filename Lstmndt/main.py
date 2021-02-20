import sys
sys.path.append('../')

from Lstmndt.detector import Detector
import argparse

parser = argparse.ArgumentParser(description='Parse path to anomaly labels if provided.')
parser.add_argument('-l', '--labels_path', default=None, required=False)
args = parser.parse_args()

if __name__ == '__main__':
    detector = Detector(labels_path=args.labels_path)

    file_name = 'SMAP/A-1.pkl'
    detector.run(file_name)
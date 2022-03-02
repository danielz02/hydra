import numpy as np
from argparse import ArgumentParser
from locuslab_smoothing.analyze import ApproximateAccuracy

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate the approximate certified accuracy")
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    approx = ApproximateAccuracy(data_file_path=args.file)
    radii = [0, 0.25, 110 / 255, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.0]
    acc = approx.at_radii(np.array(radii))
    for radius, accuracy in zip(radii, acc):
        print(f"Radius {radius}: Accuracy {accuracy}")

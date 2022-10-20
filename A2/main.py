from dataset import Dataset
from trainer import Trainer
import os
def main():
    # data = Dataset(data_file="D:/jupyter/sophia/train.json")
    # print("Dataset Created")
    t = Trainer()
    t.training()


if __name__ == '__main__':
    main()

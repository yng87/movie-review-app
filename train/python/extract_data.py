import pyprind
import pandas as pd
import os
import numpy as np

def main(basepath, output_dir, random_seed):
    """
    Download raw dataset from https://ai.stanford.edu/~amaas/data/sentiment/
    """
    labels = {"pos":1, "neg":0}

    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()

    for dir1 in ["test", "train"]:
        for dir2 in ["pos", "neg"]:
            path = os.path.join(basepath, dir1, dir2)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[dir2]]], ignore_index=True)
                pbar.update()
                
    df.columns = ["review", "sentiment"]

    np.random.seed(random_seed)
    df = df.reindex(np.random.permutation(df.index))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "movie_data.csv")
    df.to_csv(path, encoding="utf-8", index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--basepath", type=str, default='./aclImdb/')
    parser.add_argument("--output_dir", type=str, default='./csv/')

    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
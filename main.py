import torch

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import utils


def main():
    args, unknown = utils.parse_args()

    if args.embedder == 'AFGRL':
        from models import AFGRL_ModelTrainer
        embedder = AFGRL_ModelTrainer(args)

    embedder.train()
    embedder.writer.close()

if __name__ == "__main__":
    main()



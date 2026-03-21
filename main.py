import os
import torch
from tqdm import tqdm
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer 
from utils import set_seed, check_path
from configs.sid4srec_config import get_config
import os

def main():
    args = get_config()
    set_seed(args.seed) 
    check_path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    trainer.train()

if __name__ == "__main__":
    main()

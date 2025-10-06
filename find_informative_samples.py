import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from models.select_network import define_G
from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader
from utils import utils_option as option
from utils import utils_image as util

def main(args):

    # 1. Load Configuration and Trained Models
    opt = option.parse(args.opt, is_train=False)
    opt['gpu_ids'] = [0]

    # Load Bayesian Model
    netG_bayesian = define_G(opt).to('cuda')
    load_path_bayesian = args.bayesian_model_path
    print(f"Loading Bayesian model from: {load_path_bayesian}")
    netG_bayesian.load_state_dict(torch.load(load_path_bayesian), strict=True)
    netG_bayesian.eval()

    # Load Standard Model
    opt_standard = opt.copy()
    opt_standard['netG']['net_type'] = 'swinir' # Ensure we load the standard SwinIR
    netG_standard = define_G(opt_standard).to('cuda')
    load_path_standard = args.standard_model_path
    print(f"Loading Standard model from: {load_path_standard}")
    netG_standard.load_state_dict(torch.load(load_path_standard), strict=True)
    netG_standard.eval()

    # 2. Create DataLoader for the Entire Training Set
    dataset_opt = opt['datasets']['train']
    dataset_opt['dataloader_shuffle'] = False
    train_set = define_Dataset(dataset_opt)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f"Scoring {len(train_set)} images from the dataset...")

    # 3. The Active Learning Loop
    results = []
    num_samples = args.num_samples

    for data in tqdm(train_loader):
        low_res_img = data['L'].to('cuda')
        img_path = data['L_path'][0]
        img_name = os.path.basename(img_path)

        predictions = []
        with torch.no_grad():
            # Note: The standard model output can be used here for more complex
            # acquisition functions, but predictive variance only requires the Bayesian model.
            # standard_output = netG_standard(low_res_img)

            for _ in range(num_samples):
                output = netG_bayesian(low_res_img)
                output_np = util.tensor2uint(output)
                predictions.append(output_np)

        predictions_stack = np.stack(predictions, axis=0)
        variance = np.var(predictions_stack, axis=0)
        score = np.mean(variance)
        results.append((img_name, score))

    # 4. Rank and Save the Results
    results.sort(key=lambda x: x[1], reverse=True)

    output_file_path = os.path.join(opt['path']['root'], 'ranked_informativeness.txt')
    print(f"Saving ranked list to {output_file_path}")
    with open(output_file_path, 'w') as f:
        for name, score in results:
            f.write(f"{name},{score}\n")

    print("Informativeness scoring complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to your Bayesian training JSON config file.')
    parser.add_argument('--bayesian_model_path', type=str, required=True, help='Path to your trained Bayesian model (.pth) file.')
    parser.add_argument('--standard_model_path', type=str, required=True, help='Path to your trained Standard model (.pth) file.')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of Monte Carlo samples (T) to estimate uncertainty.')
    args = parser.parse_args()
    main(args)


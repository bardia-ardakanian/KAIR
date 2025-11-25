import os
import json
import random
import pandas as pd
import argparse
import logging

# --- KAIR Core Imports ---
try:
    from main_train_psnr import main as train_model_psnr
    from find_informative_samples import main as find_informative_samples
    from utils import utils_option as option
except ImportError as e:
    print("Error: Failed to import KAIR modules.")
    print(f"Please ensure this script is located in the root of the '{os.path.basename(os.getcwd())}' repository.")
    print(f"Import error: {e}")
    exit(1)


def get_config():
    """Defines all configurations for the experiment."""
    config = {
        # --- Base Config Files ---
        "base_classic_opts": "options/swinir/train_swinir_sr_classical.json",
        "base_bayesian_opts": "options/swinir/train_swinir_sr_classical_bayesian.json",

        # --- Experiment Paths & Parameters ---
        "experiment_root": "active_learning_experiment",
        
        # --- Training Data Paths ---
        "dataroot_H": "/home/ra/hniknam/Documents/Code/DIV2K_train_HR/DIV2K_train_HR",
        "dataroot_L": "/home/ra/hniknam/Documents/Code/DIV2K_train_LR_bicubic/X4",

        # --- Validation Data Paths ---
        "dataroot_H_valid": "/home/ra/hniknam/Documents/Code/DIV2K_valid_HR/DIV2K_valid_HR",
        "dataroot_L_valid": "/home/ra/hniknam/Documents/Code/DIV2K_valid_LR_bicubic/X4",

        # --- Model & Training Parameters ---
        "scale": 4, "gpu_ids": [0], "patch_size": 96, "batch_size": 8, "num_workers": 4,

        # --- Iterations ---
        "iterations_initial": 500000, "iterations_subset": 500000,

        # --- Active Learning Parameters ---
        "subset_percentages": [20, 30, 40, 50, 60, 70, 80],
        "ranked_data_file": "Informative_Data.csv",
    }
    os.makedirs(os.path.join(config["experiment_root"], "options"), exist_ok=True)
    return config

def save_option_file(cfg, opts, task_name):
    """Saves the options dict to a json file and returns the path."""
    opts_dir = os.path.join(cfg["experiment_root"], "options")
    opts_path = os.path.join(opts_dir, f"{task_name}_opts.json")
    with open(opts_path, 'w') as f:
        json.dump(opts, f, indent=2)
    return opts_path

def load_and_prepare_opts(base_opts_path, cfg, task_name):
    """Loads a base JSON config and overrides it with experiment parameters."""
    with open(base_opts_path, 'r') as f:
        opts = json.load(f)

    # Override with experiment-specific settings
    opts['task'] = task_name
    opts['scale'] = cfg['scale']
    opts['gpu_ids'] = cfg['gpu_ids']
    opts['n_channels'] = 3 # Ensure this is set for RGB

    # Paths
    opts['path']['root'] = cfg['experiment_root']
    
    # Datasets (Train)
    opts['datasets']['train']['dataroot_H'] = cfg['dataroot_H']
    opts['datasets']['train']['dataroot_L'] = cfg['dataroot_L']
    opts['datasets']['train']['H_size'] = cfg['patch_size']
    opts['datasets']['train']['dataloader_batch_size'] = cfg['batch_size']

    # Datasets (Validation/Test) - MODIFIED
    if 'test' in opts['datasets']:
        opts['datasets']['test']['dataroot_H'] = cfg['dataroot_H_valid']
        opts['datasets']['test']['dataroot_L'] = cfg['dataroot_L_valid']

    # Network
    opts['netG']['upscale'] = cfg['scale']
    opts['netG']['img_size'] = cfg['patch_size'] // cfg['scale']

    # Training iterations
    if 'train' in opts:
        opts['train']['n_iter'] = cfg['iterations_initial']

    return opts

def run_phase1_initial_training(cfg):
    """Trains the initial classic and Bayesian SwinIR models on the full dataset."""
    print("\n" + "="*80)
    print("PHASE 1: Starting Initial Model Training")
    print("="*80)

    # 1. --- Define and Train Classic SwinIR ---
    classic_task_name = "0_classic_initial"
    classic_opts = load_and_prepare_opts(cfg['base_classic_opts'], cfg, classic_task_name)
    classic_opts_path = save_option_file(cfg, classic_opts, classic_task_name)
    
    print(f"\n[PHASE 1.1] Training initial CLASSIC SwinIR model (options: {classic_opts_path})...")
    train_model_psnr(classic_opts_path)
    print("[PHASE 1.1] Initial CLASSIC model training complete.")

    # 2. --- Define and Train Bayesian SwinIR ---
    bayesian_task_name = "0_bayesian_initial"
    bayesian_opts = load_and_prepare_opts(cfg['base_bayesian_opts'], cfg, bayesian_task_name)
    bayesian_opts_path = save_option_file(cfg, bayesian_opts, bayesian_task_name)
    
    print(f"\n[PHASE 1.2] Training initial BAYESIAN SwinIR model (options: {bayesian_opts_path})...")
    train_model_psnr(bayesian_opts_path)
    print("[PHASE 1.2] Initial BAYESIAN model training complete.")


def run_phase2_uncertainty_calculation(cfg):
    """Uses the trained Bayesian model to calculate and rank data by uncertainty."""
    print("\n" + "="*80)
    print("PHASE 2: Starting Uncertainty Calculation")
    print("="*80)

    model_dir = os.path.join(cfg["experiment_root"], "0_bayesian_initial", "models")
    init_iter, bayesian_model_path = option.find_last_checkpoint(model_dir, net_type='G')
    
    if not bayesian_model_path:
        print(f"Error: Bayesian model not found in {model_dir}"); exit(1)

    finder_opts = {
        "model_path": bayesian_model_path, "dataroot_H": cfg["dataroot_H"],
        "dataroot_L": cfg["dataroot_L"], "scale": cfg["scale"]
    }
    finder_opts_path = os.path.join(cfg["experiment_root"], "options", "finder_opts.json")
    with open(finder_opts_path, 'w') as f: json.dump(finder_opts, f, indent=2)

    print(f"Calculating uncertainty using model from iteration {init_iter}: {bayesian_model_path}")
    find_informative_samples(finder_opts_path)
    
    if os.path.exists(cfg["ranked_data_file"]):
        print(f"\n[PHASE 2] Uncertainty calculation complete. Ranked data saved to '{cfg['ranked_data_file']}'.")
    else:
        print(f"Error: Expected output file '{cfg['ranked_data_file']}' was not found."); exit(1)


def run_phase3_subset_training(cfg):
    """Loops through data percentages, creates subsets, and trains new models."""
    print("\n" + "="*80)
    print("PHASE 3: Starting Subset Training Loop")
    print("="*80)

    try:
        ranked_df = pd.read_csv(cfg["ranked_data_file"])
        all_image_paths_L = ranked_df["L_path"].tolist()
    except FileNotFoundError:
        print(f"Error: Ranked data file '{cfg['ranked_data_file']}' not found."); exit(1)
    
    base_subset_opts = load_and_prepare_opts(cfg['base_classic_opts'], cfg, "dummy_task")

    for p in cfg["subset_percentages"]:
        num_to_select = int(len(all_image_paths_L) * (p / 100.0))
        print(f"\n--- Running for {p}% of data ({num_to_select} images) ---")

        active_subset_paths = all_image_paths_L[:num_to_select]
        random_subset_paths = random.sample(all_image_paths_L, num_to_select)

        subsets_dir = os.path.join(cfg["experiment_root"], "subsets")
        os.makedirs(subsets_dir, exist_ok=True)
        active_dataroot_path = os.path.join(subsets_dir, f"active_{p}_percent.txt")
        random_dataroot_path = os.path.join(subsets_dir, f"random_{p}_percent.txt")
        
        with open(active_dataroot_path, 'w') as f: f.writelines([f"{path}\n" for path in active_subset_paths])
        with open(random_dataroot_path, 'w') as f: f.writelines([f"{path}\n" for path in random_subset_paths])
            
        print(f"Created data subset files for {p}%: Active and Random")

        # Train on Active Learning subset
        active_task_name = f"active_{p}_percent"
        active_opts = json.loads(json.dumps(base_subset_opts)) 
        active_opts['task'] = active_task_name
        active_opts['datasets']['train']['dataroot_L'] = active_dataroot_path
        active_opts['train']['n_iter'] = int(cfg["iterations_subset"] * (p / 100.0))
        active_opts_path = save_option_file(cfg, active_opts, active_task_name)
        
        print(f"\n[PHASE 3.{p}.1] Training on {p}% ACTIVE subset...")
        train_model_psnr(active_opts_path)
        print(f"[PHASE 3.{p}.1] ACTIVE subset training complete.")
        
        # Train on Random subset
        random_task_name = f"random_{p}_percent"
        random_opts = json.loads(json.dumps(base_subset_opts)) 
        random_opts['task'] = random_task_name
        random_opts['datasets']['train']['dataroot_L'] = random_dataroot_path
        random_opts['train']['n_iter'] = int(cfg["iterations_subset"] * (p / 100.0))
        random_opts_path = save_option_file(cfg, random_opts, random_task_name)
        
        print(f"\n[PHASE 3.{p}.2] Training on {p}% RANDOM subset...")
        train_model_psnr(random_opts_path)
        print(f"[PHASE 3.{p}.2] RANDOM subset training complete.")

    print("\n[PHASE 3] Subset training loop finished.")


def main():
    parser = argparse.ArgumentParser(description="Run a full active learning experiment for SwinIR.")
    parser.add_argument('--skip_phase1', action='store_true', help="Skip initial model training (Phase 1).")
    parser.add_argument('--skip_phase2', action='store_true', help="Skip uncertainty calculation (Phase 2).")
    parser.add_argument('--skip_phase3', action='store_true', help="Skip subset training (Phase 3).")
    args = parser.parse_args()

    config = get_config()
    logging.getLogger('train').setLevel(logging.WARNING)

    if not args.skip_phase1:
        run_phase1_initial_training(config)
    else:
        print("Skipping Phase 1: Initial Model Training.")

    if not args.skip_phase2:
        run_phase2_uncertainty_calculation(config)
    else:
        print("Skipping Phase 2: Uncertainty Calculation.")

    if not args.skip_phase3:
        run_phase3_subset_training(config)
    else:
        print("Skipping Phase 3: Subset Training.")
        
    print("\n" + "="*80)
    print("Active Learning Experiment Script Finished.")
    print(f"All outputs can be found in the '{config['experiment_root']}' directory.")
    print("Next step: Use 'main_test_swinir.py' to evaluate the trained models on a validation set.")


if __name__ == '__main__':
    main()
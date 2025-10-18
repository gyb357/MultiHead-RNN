import pandas as pd
import torch
from yaml import safe_load
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from model.model import MultiHead, SingleHead
from seed import make_seed_list, derive_seed, set_seed, seed_worker
from dataset import (
    undersample_dataset,
    processing,
    extract_variable_train,
    to_tensor_dataset
)
from typing import List
from torch.utils.data import DataLoader
from model.utils import get_parameters
from train.train import fit, eval
from train.utils import save_results


# Load configuration
with open('config/configs.yaml', 'r') as file:
    config = safe_load(file)


# Scaler and model mappings
_scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'minmax': MinMaxScaler()
}
_head = {
    'multi': MultiHead,
    'single': SingleHead
}


# Hyperparameters
RUNS = config['runs']
WINDOW_START = config['window_start']
WINDOW_END = config['window_end']
BAL_TRAIN = config['bal_train']
BAL_VALID = config['bal_valid']
BAL_TEST = config['bal_test']
SCALER = _scalers[config['scaler']]
BATCH_SIZE = config['batch_size']
DEVICE = config['device']
MODEL = _head[config['head']]
TYPE = config['type']
RNN_HIDDEN_SIZE = config['rnn_hidden_size']
FC_HIDDEN_SIZE = config['fc_hidden_size']
NUM_CLASSES = config['num_classes']
LR = config['lr']
EPOCHS = config['epochs']
PATIENCE = config['patience']
THRESHOLD = config['threshold']

# Seed
SEED_MASTER = config['seed_master']
DETERMINISTIC = config['deterministic']
SEED_LIST = make_seed_list(RUNS, SEED_MASTER)


# Main loop
if __name__ == '__main__':
    for run in range(1, RUNS + 1):
        base_seed = SEED_LIST[run - 1]

        for window in range(WINDOW_START, WINDOW_END + 1):
            seed = derive_seed(base_seed, window)
            set_seed(seed, deterministic=DETERMINISTIC)

            print(f"\n[Run: {run} | Window: {window}] | Seed: {seed}")

            # Load datasets
            train_df = pd.read_csv(f'dataset/{window}_train.csv')
            valid_df = pd.read_csv(f'dataset/{window}_valid.csv')
            test_df = pd.read_csv(f'dataset/{window}_test.csv')

            # Preprocess datasets (optional undersampling)
            if BAL_TRAIN: train_df = undersample_dataset(train_df, window)  # BAL_TRAIN = True
            if BAL_VALID: valid_df = undersample_dataset(valid_df, window)  # BAL_VALID = False
            if BAL_TEST: test_df = undersample_dataset(test_df, window)     # BAL_TEST = False

            # CIK and status for evaluation
            cik_status = test_df[['cik', 'status']].copy()
            num_samples = len(cik_status) // window * window
            cik_status_df = (cik_status.iloc[0:num_samples:window])
            # [0:num_samples:window] = 0, window, 2 * window, ..., (n-1) * window

            # Feature-target split
            x_train, y_train = processing(train_df)
            x_valid, y_valid = processing(valid_df)
            x_test, y_test = processing(test_df)
            
            # Variable extraction
            variables = x_train.columns.tolist()

            # Feature scaling
            x_train_scaled = SCALER.fit_transform(x_train.values)
            x_valid_scaled = SCALER.transform(x_valid.values)
            x_test_scaled = SCALER.transform(x_test.values)

            # Convert back to DataFrame for easier variable extraction
            x_train_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
            x_valid_df = pd.DataFrame(x_valid_scaled, columns=x_valid.columns)
            x_test_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

            # Extract each variable's time series data
            x_train_list: List = []
            x_valid_list: List = []
            for v in variables:
                x_train_list.append(extract_variable_train(x_train_df, v, window))
                x_valid_list.append(extract_variable_train(x_valid_df, v, window))
            x_test_list: List = [extract_variable_train(x_test_df, var, window) for var in variables]

            # Create TensorDatasets
            train_dataset = to_tensor_dataset(x_train_list, y_train, window)
            valid_dataset = to_tensor_dataset(x_valid_list, y_valid, window)
            test_dataset = to_tensor_dataset(x_test_list, y_test, window)

            # Integrity check
            expected_test_samples = len(test_dataset)
            actual_dataset_samples = len(cik_status_df)
            print(f"Expected test samples: {expected_test_samples}, Actual dataset samples: {actual_dataset_samples}")

            if expected_test_samples != actual_dataset_samples:
                raise ValueError("Mismatch between expected test samples and actual dataset samples")

            # Dataset summaries
            print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
            print(f"CIK status entries: {len(cik_status_df)}")

            # DataLoaders
            g_train = torch.Generator()
            g_train.manual_seed(seed)
            g_eval = torch.Generator()
            g_eval.manual_seed(seed)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g_train
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g_eval
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g_eval
            )

            # Device setup
            device = torch.device(DEVICE)
            print(f"Using device: {device}")

            model = MODEL(
                rnn_hidden_size=RNN_HIDDEN_SIZE,
                num_variables=len(variables),
                fc_hidden_size=FC_HIDDEN_SIZE,
                num_classes=NUM_CLASSES,
                rnn_type=TYPE
            ).to(device)

            # Model summary
            print(f"RNN Parameters: {get_parameters(model.rnn)}")
            print(f"Projection Parameters: {get_parameters(model.proj)}")
            print(f"FC Parameters: {get_parameters(model.fc)}")
            print(f"Total Parameters: {get_parameters(model)}")
            
            # Logging path
            csv_path = f"result/{MODEL.__name__}_{TYPE.upper()}_{SCALER.__class__.__name__}_{THRESHOLD}_rnnsize-{RNN_HIDDEN_SIZE}_fcsize-{FC_HIDDEN_SIZE}"
            
            # Training
            train_time = fit(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=device,
                lr=LR,
                epochs=EPOCHS,
                patience=PATIENCE,
                threshold=THRESHOLD,
                run=run,
                window=window,
                csv_path=csv_path + "_valid.csv"
            )

            # Load best model for evaluation
            checkpoint = torch.load('result/best_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model'])

            # Check dataset lengths
            if len(test_dataset) != len(cik_status_df):
                print(f"Critical Error: Dataset length mismatch!")
                print(f" - Test dataset: {len(test_dataset)} samples")
                print(f" - Expected from CIK data: {len(cik_status_df)} samples")
                continue

            metrics = eval(
                model=model,
                data_loader=test_loader,
                device=device,
                threshold=THRESHOLD,
                cik_status=cik_status_df,
                csv_path=csv_path + f"_window-{window}_preds.csv"
            )
            # Log metrics
            print(f"[Window {window}] "
                  f"Test Accuracy = {metrics[0]:.4f}, "
                  f"AUC = {metrics[1]:.4f}, "
                  f"BAC = {metrics[2]:.4f}")
            
            # Save predictions
            save_results(
                run=run,
                window=window,
                metrics=metrics,
                train_time=train_time,
                csv_path=csv_path + "_test.csv"
            )

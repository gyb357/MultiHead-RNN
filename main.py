import hydra
import pandas as pd
import torch
import time
from omegaconf import DictConfig
from seed import make_seed_list, derive_seed, set_seed, seed_worker
from utils import get_scaler, get_model_class, get_parameters
from dataset.dataset import undersampling, separate_labels, extract_variable_train, to_tensor_dataset
from torch.utils.data import DataLoader
from train.train import fit, eval, save_results


_CONFIG_PATH = 'config'
_CONFIG_NAME = 'config'


# Main function
@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name=_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    # Setup
    seed_list = make_seed_list(cfg.runs, cfg.seed_master)  # Seed configuration
    scaler = get_scaler(cfg.scaler)                        # Scaler type
    model_type = get_model_class(cfg.head)                 # Model type

    # Main loop
    for run in range(1, cfg.runs + 1):
        base_seed = seed_list[run - 1]

        for window in range(cfg.window_start, cfg.window_end + 1):
            # Set seed
            seed = derive_seed(base_seed, window)
            set_seed(seed, cfg.deterministic)

            # Logging
            print(f"\n[Run: {run} | Window: {window}] | Seed: {seed}")

            # Load datasets
            train_df = pd.read_csv(f'dataset/{window}_train.csv')
            valid_df = pd.read_csv(f'dataset/{window}_valid.csv')
            test_df = pd.read_csv(f'dataset/{window}_test.csv')

            # Undersampling
            if cfg.undersample_train: train_df = undersampling(train_df, window)  # True
            if cfg.undersample_valid: valid_df = undersampling(valid_df, window)  # False
            if cfg.undersample_test: test_df = undersampling(test_df, window)     # False

            # CIK and status for evaluation
            cik_status = test_df[['cik', 'status']].copy()
            total_samples = len(cik_status)

            num_valid_samples = (total_samples // window) * window
            cik_status_df = cik_status.iloc[:num_valid_samples:window].reset_index(drop=True)
            # [:num_samples:window] = 0, window, 2 * window, ..., (n-1) * window

            # Feature & target separation
            x_train, y_train = separate_labels(train_df)
            x_valid, y_valid = separate_labels(valid_df)
            x_test, y_test = separate_labels(test_df)

            # Variable extraction
            variables = x_train.columns.tolist()

            # Scaling
            x_train_scaled = scaler.fit_transform(x_train.values)
            x_valid_scaled = scaler.transform(x_valid.values)
            x_test_scaled = scaler.transform(x_test.values)

            # Convert back to dataframe
            x_train_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
            x_valid_df = pd.DataFrame(x_valid_scaled, columns=x_valid.columns)
            x_test_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

            # Extract variables
            x_train_list = [extract_variable_train(x_train_df, v, window) for v in variables]
            x_valid_list = [extract_variable_train(x_valid_df, v, window) for v in variables]
            x_test_list = [extract_variable_train(x_test_df, v, window) for v in variables]

            # Create TensorDatasets
            train_dataset = to_tensor_dataset(x_train_list, y_train, window)
            valid_dataset = to_tensor_dataset(x_valid_list, y_valid, window)
            test_dataset = to_tensor_dataset(x_test_list, y_test, window)


            # DataLoaders with proper seeding
            g_train = torch.Generator()
            g_train.manual_seed(seed)
            g_eval = torch.Generator()
            g_eval.manual_seed(seed)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g_train
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g_eval
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g_eval
            )

            # Device setup
            device = torch.device(cfg.device)

            # Model initialization
            model = model_type(
                num_variables=len(variables),
                rnn_hidden_size=cfg.rnn_hidden_size,
                projection_size=cfg.projection_size,
                fc_hidden_size=cfg.fc_hidden_size,
                num_classes=cfg.num_classes,
                rnn_type=cfg.type,
                t_max=window,
                dropout=cfg.dropout,
                **cfg.rnn_kwargs
            ).to(device)

            # Model summary
            print(f"RNN Parameters: {get_parameters(model.rnn)}")
            print(f"Projection Parameters: {get_parameters(model.projection)}")
            print(f"FC Parameters: {get_parameters(model.fc)}")
            print(f"Total Parameters: {get_parameters(model)}")

            # Logging path
            csv_path = (
                f"result/"
                f"{model_type.__name__}"
                f"{cfg.type.upper()}_"
                f"{scaler.__class__.__name__}_"
                f"{cfg.threshold}_"        # Threshold
                f"{cfg.rnn_hidden_size}_"  # RNN size
                f"{cfg.fc_hidden_size}"    # FC size
            )

            # Training
            fit(
                model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
                epochs=cfg.epochs,
                lr=cfg.lr,
                patience=cfg.patience,
                accumulation_steps=cfg.accumulation_steps,
                num_classes=cfg.num_classes,
                threshold=cfg.threshold,
                run=run,
                window=window,
                csv_path=csv_path + "_valid.csv"
            )

            # Load best model for evaluation
            checkpoint = torch.load('result/best_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model'])

            # Test evaluation
            start_time = time.time()
            test_metrics = eval(
                model=model,
                data_loader=test_dataloader,
                device=device,
                threshold=cfg.threshold,
                cik_status=cik_status_df,
                csv_path=csv_path + f"_window-{window}_preds.csv"
            )
            end_time = time.time()

            # Save predictions
            save_results(
                run,
                window=window,
                metrics=test_metrics,
                train_time=end_time - start_time,
                csv_path=csv_path + "_test.csv"
            )


if __name__ == '__main__':
    main()


import torch.nn as nn
import torch
import time
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from typing import List, Optional, Dict
from torch import Tensor
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from train.utils import save_results, probability
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)


# Progress bar metric name
METRIC_NAME = "PR-AUC"


def fit(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        lr: float,
        epochs: int,
        patience: int,
        threshold: float,
        run: int,
        window: int,
        csv_path: str
) -> float:
    # Compute class weights
    labels: List[Tensor] = []

    for batch in train_loader:
        *_, y = batch
        labels.append(y.detach().cpu().long())

    labels = torch.cat(labels, dim=0)
    counts = torch.bincount(labels, minlength=2).float()
    inv_freq = 1.0 / (counts + 1e-8)
    class_weights = (inv_freq / inv_freq.sum()).to(device)

    # Define torch train components
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=patience // 2,
    )

    # Training state
    best_metric: float = -float("inf")
    best_metrics: Optional[List[float]] = None
    wait: int = 0
    train_times: List[float] = []
    avg_train_time: float = 0.0

    # Training loop with progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        refresh_per_second=10,
    ) as progress:
        # Initialize progress bar
        task = progress.add_task(f"Epoch 1/{epochs} • {METRIC_NAME}: 0.0000", total=len(train_loader))

        for epoch in range(1, epochs + 1):
            # Reset progress bar for new epoch
            progress.reset(
                task,
                total=len(train_loader),
                completed=0,
                description=f"Epoch {epoch}/{epochs} • {METRIC_NAME}: {max(0.0, best_metric):.4f}"
            )

            model.train()
            total_loss: float = 0.0
            train_start_time = time.time()

            # Training batches
            for batch in train_loader:
                *x, y = batch
                x = [xi.to(device) for xi in x]
                y = y.to(device)

                # Initialize gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = model(x)
                # Compute loss
                loss = criterion(outputs, y)

                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                progress.advance(task)

            # End of epoch
            train_times.append(time.time() - train_start_time)
            avg_train_time = sum(train_times) / len(train_times)
            train_loss = total_loss / len(train_loader.dataset)

            # Validation
            valid_metrics = eval(
                model=model,
                data_loader=valid_loader,
                device=device,
                threshold=threshold,
                cik_status=None,
                csv_path=None
            )
            valid_metric_raw = valid_metrics[3] # PR-AUC-1 (average_precision_score)

            # Scheduler step
            scheduler.step(valid_metric_raw)

            # Early stopping check
            if valid_metric_raw > best_metric:
                best_metric = valid_metric_raw
                best_metrics = valid_metrics
                wait = 0

                # Save best model
                os.makedirs('result', exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, 'result/best_model.pth')
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    # Save final results
                    if best_metrics is not None:
                        save_results(
                            run=run,
                            window=window,
                            metrics=best_metrics,
                            train_time=avg_train_time,
                            csv_path=csv_path,
                        )
                    return avg_train_time

    # Final save after all epochs
    if best_metrics is not None:
        save_results(
            run=run,
            window=window,
            metrics=best_metrics,
            train_time=avg_train_time,
            csv_path=csv_path,
        )
    return avg_train_time


def eval(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        threshold: float,
        cik_status: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None
) -> List[float]:
    model.eval()

    # Evaluation state
    probs: List[float] = []
    labels: List[int] = []

    # For company-level aggregation
    company_predictions: Dict[str, Dict[str, int]] = {}  # {"cik(str)": {"actual_label": int, "pred_0_count": int, "pred_1_count": int}}
    # Offset of batch samples in the entire dataset
    cursor: int = 0
    # Arrays to store cik
    ciks_arr: Optional[np.ndarray] = None
    acts_arr: Optional[np.ndarray] = None

    # Setup cik arrays
    if cik_status is not None:
        if 'cik' not in cik_status.columns:
            raise ValueError("cik_status DataFrame must contain 'cik' column.")
        # Convert 'cik' column to string type and then to numpy array
        ciks_arr = cik_status['cik'].astype(str).to_numpy()

        if 'status' in cik_status.columns:
            acts_arr = cik_status['status'].to_numpy().astype(int)

        expected_len = len(data_loader.dataset)
        if len(ciks_arr) != expected_len:
            raise ValueError(f"cik_status length {len(ciks_arr)} does not match dataset length {expected_len}")

    mismatches: int = 0  # Count of cik mismatches

    # Evaluation loop
    with torch.no_grad():
        for batch in data_loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            y = y.to(device)

            # Compute logits and probabilities
            logits = model(x)
            prob = F.softmax(logits, dim=1)[:, 1]

            # Company-level predictions
            batch_probs = prob.detach().cpu().numpy()
            batch_labels = y.detach().cpu().numpy().astype(int)
            batch_preds = probability(batch_probs, threshold)  # 0/1

            probs.extend(batch_probs.tolist())
            labels.extend(batch_labels.tolist())

            # Company-level aggregation
            if ciks_arr is not None:
                batch_size = len(batch_labels)
                # Current batch's cik and actual status slices
                ciks_slice = ciks_arr[cursor:cursor + batch_size]
                acts_slice = acts_arr[cursor:cursor + batch_size] if acts_arr is not None else batch_labels

                # Check for mismatches in length
                if acts_arr is not None:
                    mismatches += int(np.sum(acts_slice != batch_labels))

                # per-CIK
                for cik_raw, act, pred in zip(ciks_slice, acts_slice, batch_preds):
                    cik = str(cik_raw)
                    rec = company_predictions.get(cik)
                    if rec is None:
                        company_predictions[cik] = {
                            'actual_label': int(act),
                            'pred_0_count': 1 if pred == 0 else 0,
                            'pred_1_count': 1 if pred == 1 else 0
                        }
                    else:
                        # Warn if actual label mismatch
                        if rec['actual_label'] != int(act):
                            print(f"[Warn] CIK {cik}: multiple actual labels detected "
                                  f"({rec['actual_label']} vs {int(act)}). Using the first.")
                        if pred == 0:
                            rec['pred_0_count'] += 1
                        elif pred == 1:
                            rec['pred_1_count'] += 1

                # Move cursor
                cursor += batch_size

    if ciks_arr is not None and cursor != len(ciks_arr):
        print(f"[Warn] Cursor/end mismatch: consumed {cursor} rows but cik_status has {len(ciks_arr)} rows.")
    if mismatches > 0:
        print(f"[Warn] There were {mismatches} mismatches between provided actual statuses and batch labels.")

    if company_predictions and (csv_path is not None):
        rows: List[dict] = []
        for cik, data in company_predictions.items():
            rows.append({
                'cik': str(cik),
                'actual_status': int(data['actual_label']),
                'pred_0_count': int(data['pred_0_count']),
                'pred_1_count': int(data['pred_1_count']),
                'total_predictions': int(data['pred_0_count'] + data['pred_1_count'])
            })
        df_new = pd.DataFrame(rows)
        df_new['cik'] = df_new['cik'].astype(str)

        try:
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path, dtype={'cik': str})
                df_old['cik'] = df_old['cik'].astype(str)
                df_concat = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_concat = df_new

            # groupby cik and aggregate
            df_concat['cik'] = df_concat['cik'].astype(str)
            aggregation_rules = {
                'actual_status': 'first',
                'pred_0_count': 'sum',
                'pred_1_count': 'sum',
                'total_predictions': 'sum'
            }
            df_final = df_concat.groupby('cik', as_index=False, sort=False).agg(aggregation_rules)
            df_final.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[Error] while reading or writing CSV: {e}")

    # Convert to numpy arrays
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    # Compute binary predictions
    preds = probability(probs, threshold)

    # Standard metrics
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    bac = balanced_accuracy_score(labels, preds)
    # PR-AUC (average precision score)
    pr_auc1 = average_precision_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc2 = auc(recall, precision)
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    # Class-specific recall and precision
    rec_bankruptcy = recall_score(labels, preds, pos_label=1, zero_division=0)
    pr_bankruptcy  = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec_healthy = recall_score(labels, preds, pos_label=0, zero_division=0)
    pr_healthy = precision_score(labels, preds, pos_label=0, zero_division=0)
    # F1 scores
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    # Calculate type I and type II errors
    type_1_error = 1.0 - rec_healthy    # = fp / (fp + tn)
    type_2_error = 1.0 - rec_bankruptcy # = fn / (tp + fn)

    return (
        acc, roc_auc, bac,
        pr_auc1, pr_auc2,
        tn, fp, fn, tp,
        micro_f1, macro_f1,
        type_1_error, type_2_error,
        rec_bankruptcy, pr_bankruptcy, rec_healthy, pr_healthy
    )


from . import *


# Configuration for training
_CONFIG = {
    # Scheduler
    'scheduler_mode': 'max',
    'scheduler_factor': 0.5,

    # Metric name for display
    'metric_name': 'BAC + PR-AUC'
}


def fit(
        # Training
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        device: device,
        epochs: int,
        lr: float,
        patience: int,
        accumulation_steps: int,
        num_classes: int,
        # Evaluation
        threshold: float,
        # Saving
        run: int,
        window: int,
        csv_path: str
) -> None:
    # Compute class weights
    labels = []
    for batch in train_dataloader:
        *_, y = batch
        labels.append(y.detach().cpu().long())

    labels = torch.cat(labels, dim=0)

    # Training components
    criterion = nn.CrossEntropyLoss(class_weights(labels, num_classes, device))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=_CONFIG['scheduler_mode'],
        factor=_CONFIG['scheduler_factor'],
        patience=patience // 2,
    )
    early_stopping = EarlyStopping(patience, mode=_CONFIG['scheduler_mode'])

    # Training state
    train_times = []
    best_metrics = None
    best_metric = 0.0

    # Training loop with progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        refresh_per_second=10,
    ) as progress:
        
        # Initialize progress bar
        task = progress.add_task(f"Epoch 1/{epochs} • {_CONFIG['metric_name']}: 0.0000", total=len(train_dataloader))

        for epoch in range(1, epochs + 1):
            # Reset progress bar for new epoch
            progress.reset(
                task,
                total=len(train_dataloader),
                completed=0,
                description=f"Epoch {epoch}/{epochs} • {_CONFIG['metric_name']}: {best_metric:.4f}"
            )

            model.train()
            train_time = time.time()

            # Training batches
            for batch_idx, batch in enumerate(train_dataloader):
                *x, y = batch
                x = [xi.to(device) for xi in x]
                y = y.to(device)

                # Forward pass
                outputs = model(x)
                loss = criterion(outputs, y)

                # Divide loss by accumulation steps
                loss = loss / accumulation_steps

                # Backward pass
                loss.backward()

                # Optimization step
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer_step(model, optimizer)

                # Update progress bar
                progress.advance(task)

            # Final optimization step
            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer_step(model, optimizer)

            # Training time
            train_times.append(time.time() - train_time)
            avg_train_time = sum(train_times) / len(train_times)

            # Validation
            valid_metrics: ClassificationMetrics = eval(
                model=model,
                data_loader=valid_dataloader,
                device=device,
                threshold=threshold,
                cik_status=None,
                csv_path=None
            )

            # Early stopping
            current_metric = valid_metrics.balanced_accuracy + valid_metrics.pr_auc

            # Save best model
            if current_metric > best_metric:
                best_metrics = valid_metrics
                best_metric = current_metric

                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, 'result/best_model.pth')
                
            if early_stopping(current_metric):
                break

            # Step scheduler
            scheduler.step(current_metric)

    # Save final results
    if best_metrics is not None:
        save_results(
            run=run,
            window=window,
            metrics=best_metrics,
            train_time=avg_train_time,
            csv_path=csv_path
        )


def eval(
        # Evaluation
        model: nn.Module,
        data_loader: DataLoader,
        device: device,
        threshold: float,

        # Company-level aggregation
        cik_status: Optional[pd.DataFrame],

        # Saving
        csv_path: Optional[str]
) -> ClassificationMetrics:
    model.eval()

    # Evaluation state
    probs = []
    labels = []

    # Company-level aggregation
    company_predictions = {}
    cursor = 0
    ciks_arr = None
    acts_arr = None

    # Setup cik arrays
    if cik_status is not None:
        if 'cik'    not in cik_status.columns or \
           'status' not in cik_status.columns:
            raise ValueError("'cik_status' DataFrame must contain 'cik' and 'status' columns.")

        ciks_arr = cik_status['cik'].to_numpy()
        acts_arr = cik_status['status'].to_numpy().astype(int)

        expected_len = len(data_loader.dataset)
        if len(ciks_arr) != expected_len:
            raise ValueError(f"'cik_status' length {len(ciks_arr)} does not match dataset length {expected_len}.")
        
    # Evaluation batches
    with torch.no_grad():
        for batch in data_loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            y = y.to(device)

            # Forward pass
            outputs = model(x)

            # Probabilities
            prob = F.softmax(outputs, dim=1)[:, 1]

            batch_probs = detach(prob).tolist()
            batch_labels = detach(y).astype(int)
            batch_preds = probability(batch_probs, threshold)

            # Store batch results
            probs.extend(batch_probs)
            labels.extend(batch_labels)

            # Company-level aggregation
            if cik_status is not None:
                batch_size = len(y)
                ciks_slice = ciks_arr[cursor:cursor + batch_size]
                acts_slice = acts_arr[cursor:cursor + batch_size]

                for cik, act, pred in zip(ciks_slice, acts_slice, batch_preds):
                    cik = str(cik)
                    rec = company_predictions.get(cik)

                    if rec is None:
                        company_predictions[cik] = {
                            'label': int(act),
                            'pred_0': 1 if pred == 0 else 0,
                            'pred_1': 1 if pred == 1 else 0
                        }
                    else:
                        rec['pred_0'] += 1 if pred == 0 else 0
                        rec['pred_1'] += 1 if pred == 1 else 0

                # Move cursor
                cursor += batch_size

    # Save company-level predictions
    if company_predictions and (csv_path is not None):
        rows = []

        for cik, rec in company_predictions.items():
            rows.append({
                'cik': cik,
                'label': rec['label'],
                'pred_0': rec['pred_0'],
                'pred_1': rec['pred_1'],
                'total': rec['pred_0'] + rec['pred_1']
            })
            
        # Create DataFrame
        df_new = pd.DataFrame(rows)
        df_new['cik'] = df_new['cik'].astype(str)

        # Concat with old results
        try:
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path, dtype={'cik': str})
                df_old['cik'] = df_old['cik'].astype(str)

                df_concat = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_concat = df_new

            # Groupby cik and aggregate
            df_concat['cik'] = df_concat['cik'].astype(str)
            
            aggregation_rules = {
                'label': 'first',
                'pred_0': 'sum',
                'pred_1': 'sum',
                'total': 'sum'
            }
            df_final = df_concat.groupby('cik', as_index=False, sort=False).agg(aggregation_rules)
            df_final.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error saving company-level predictions: {e}")

    return binary_classification_metrics(
        probabilities=probs,
        labels=labels,
        threshold=threshold
    )


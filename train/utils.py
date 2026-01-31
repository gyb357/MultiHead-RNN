from . import *


class EarlyStopping():
    def __init__(
            self,
            patience: int,
            mode: str
    ) -> None:
        # Attributes
        self.patience = patience
        self.mode = mode

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode should be 'min' or 'max', got '{mode}'.")
        
        # Internal variables
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _update(self, score: float) -> None:
        self.best_score = score
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self._update(score)

            return self.early_stop
        
        improved = (score > self.best_score) if self.mode == 'max' else (score < self.best_score)
        if improved:
            self._update(score)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class ClassificationMetrics(NamedTuple):
    # Confusion matrix
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    # Overall metrics
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float

    # F1 scores
    micro_f1: float
    macro_f1: float

    # Class-specific metrics
    recall_bankruptcy: float
    precision_bankruptcy: float
    recall_healthy: float
    precision_healthy: float

    # Error rates
    type_1_error: float
    type_2_error: float

    def to_list(self) -> List[Union[int, float]]:
        return [
            # Confusion matrix
            self.true_positive, self.true_negative, self.false_positive, self.false_negative,
            # Overall metrics
            self.accuracy, self.balanced_accuracy, self.roc_auc, self.pr_auc,
            # F1 scores
            self.micro_f1, self.macro_f1,
            # Class-specific metrics
            self.recall_bankruptcy, self.precision_bankruptcy, self.recall_healthy, self.precision_healthy,
            # Error rates
            self.type_1_error, self.type_2_error
        ]
    
    def to_dict(self) -> Dict[str, Union[int, float]]:
        return self._asdict()


def class_weights(
        labels: Tensor,
        num_classes: int,
        device: device
) -> Tensor:
    # Inverse frequency weighting
    count = torch.bincount(labels, minlength=num_classes).float()
    inv_freq = 1.0 / (count + 1e-8)
    weights = (inv_freq / inv_freq.sum()).to(device)

    return weights


def optimizer_step(
        model: nn.Module,
        optimizer: optim.Optimizer,
        max_norm: float = 1.0
) -> None:
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    optimizer.zero_grad()


def save_results(
        run: int,
        window: int,
        metrics: ClassificationMetrics,
        train_time: float,
        csv_path: str
) -> None:
    metrics_dict = metrics.to_dict()
    metrics_dict.update({
        'run': run,
        'window': window,
        'train_time': train_time
    })
    results = pd.DataFrame([metrics_dict])

    # Write header
    write_header = (run == 1 and window == 3)

    # Ensure directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    results.to_csv(
        csv_path,
        mode='a',
        index=False,
        header=write_header,
        float_format='%.4f',
        lineterminator='\n'
    )


def detach(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def probability(
        probabilities: List[float],
        threshold: float = 0.5
) -> List[int]:
    return [1 if prob >= threshold else 0 for prob in probabilities]


def binary_classification_metrics(
        probabilities: List[float],
        labels: List[int],
        threshold: float = 0.5
) -> ClassificationMetrics:
    prob = np.asarray(probabilities)
    lab = np.asarray(labels)

    # Binary predictions
    pred = (prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(lab, pred, labels=[0, 1]).ravel()

    # Overall metrics
    acc = accuracy_score(lab, pred)
    bac = balanced_accuracy_score(lab, pred)
    roc_auc = roc_auc_score(lab, prob)
    pr_auc = average_precision_score(lab, prob)

    # F1 scores
    micro_f1 = f1_score(lab, pred, average='micro', zero_division=0)
    macro_f1 = f1_score(lab, pred, average='macro', zero_division=0)

    # Class-specific metrics
    rec_bankruptcy = recall_score(  lab, pred, pos_label=1, zero_division=0)
    pr_bankruptcy = precision_score(lab, pred, pos_label=1, zero_division=0)
    rec_healthy = recall_score(     lab, pred, pos_label=0, zero_division=0)
    pr_healthy = precision_score(   lab, pred, pos_label=0, zero_division=0)

    # Error rates
    type_1_err = 1.0 - rec_healthy    # FPR = 1 - Specificity
    type_2_err = 1.0 - rec_bankruptcy # FNR = 1 - Sensitivity

    return ClassificationMetrics(
        tp, tn, fp, fn,
        acc, bac, roc_auc, pr_auc,
        micro_f1, macro_f1,
        rec_bankruptcy, pr_bankruptcy, rec_healthy, pr_healthy,
        type_1_err, type_2_err
    )


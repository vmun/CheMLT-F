"""
Load pretrained BindingDB BALM, fine-tune on Davis/KIBA train split,
then evaluate on the test split.

"""

import argparse
import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from balm import common_utils
from balm.configs import Configs
from balm.models import BALM
from balm.models.utils import load_trained_model
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col_map = {
        "compound_iso_smiles": "Drug",
        "target_sequence": "Target",
        "affinity": "Y",
    }
    df = df.rename(columns=col_map)
    required = {"Drug", "Target", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def pkd_to_cosine(y, pkd_lower_bound, pkd_upper_bound):
    """Scale pKd label to [-1, 1] cosine similarity range."""
    return (y - pkd_lower_bound) / (pkd_upper_bound - pkd_lower_bound) * 2 - 1


def finetune(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_tokenizer,
    drug_tokenizer,
    pkd_lower_bound: float,
    pkd_upper_bound: float,
    device: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    protein_max_len: int,
    drug_max_len: int,
    save_model_path: str = None,
    eval_every: int = 5,
):
    """Fine-tune only the projection head (same as notebook pattern)."""

    # Freeze everything except projection layer — same as load_trained_model(is_training=True)
    for name, param in model.named_parameters():
        if "projection" not in name:
            param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Fine-tuning {sum(p.numel() for p in trainable):,} parameters (projection only)")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()

    drugs   = train_df["Drug"].tolist()
    targets = train_df["Target"].tolist()
    ys      = train_df["Y"].tolist()
    
    print("\nRunning evaluation at epoch 0 (before fine-tuning)...")
    predictions, labels = run_inference(
        model=model,
        df=test_df,
        protein_tokenizer=protein_tokenizer,
        drug_tokenizer=drug_tokenizer,
        pkd_lower_bound=pkd_lower_bound,
        pkd_upper_bound=pkd_upper_bound,
        device=device,
        batch_size=batch_size,
        protein_max_len=protein_max_len,
        drug_max_len=drug_max_len,
    )
    compute_and_print_metrics(labels, predictions, "validation")
    model.train() 
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for start in tqdm(range(0, len(train_df), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_drugs   = drugs[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            batch_ys      = ys[start : start + batch_size]

            protein_inputs = protein_tokenizer(
                batch_targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=protein_max_len,
            ).to(device)

            drug_inputs = drug_tokenizer(
                batch_drugs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=drug_max_len,
            ).to(device)

            # Scale labels to cosine similarity range
            cosine_labels = torch.tensor(
                [pkd_to_cosine(y, pkd_lower_bound, pkd_upper_bound) for y in batch_ys],
                dtype=torch.float32,
            ).to(device)

            inputs = {
                "protein_input_ids":      protein_inputs["input_ids"],
                "protein_attention_mask": protein_inputs["attention_mask"],
                "drug_input_ids":         drug_inputs["input_ids"],
                "drug_attention_mask":    drug_inputs["attention_mask"],
                "labels":                 cosine_labels,
            }

            optimizer.zero_grad()
            loss = model(inputs)["loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.4f}")

        if (epoch + 1) % eval_every == 0:
            print(f"\nRunning evaluation at epoch {epoch+1}...")
    
            predictions, labels = run_inference(
                model=model,
                df=test_df,
                protein_tokenizer=protein_tokenizer,
                drug_tokenizer=drug_tokenizer,
                pkd_lower_bound=pkd_lower_bound,
                pkd_upper_bound=pkd_upper_bound,
                device=device,
                batch_size=batch_size,
                protein_max_len=protein_max_len,
                drug_max_len=drug_max_len,
            )
    
            compute_and_print_metrics(labels, predictions, "validation")

            if save_model_path:
                base, ext = os.path.splitext(save_model_path)
                epoch_path = f"{base}_epoch{epoch+1}{ext}"
                torch.save(model.state_dict(), epoch_path)
                print(f"Checkpoint saved to {epoch_path}")

            model.train()  # restore train mode after eval

    return model


@torch.no_grad()
def run_inference(
    model,
    df: pd.DataFrame,
    protein_tokenizer,
    drug_tokenizer,
    pkd_lower_bound: float,
    pkd_upper_bound: float,
    device: str,
    batch_size: int,
    protein_max_len: int,
    drug_max_len: int,
):
    model.eval()
    predictions = []
    labels = []

    drugs   = df["Drug"].tolist()
    targets = df["Target"].tolist()
    ys      = df["Y"].tolist()

    for start in tqdm(range(0, len(df), batch_size), desc="Inference"):
        batch_drugs   = drugs[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        batch_ys      = ys[start : start + batch_size]

        protein_inputs = protein_tokenizer(
            batch_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=protein_max_len,
        ).to(device)

        drug_inputs = drug_tokenizer(
            batch_drugs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=drug_max_len,
        ).to(device)

        inputs = {
            "protein_input_ids":      protein_inputs["input_ids"],
            "protein_attention_mask": protein_inputs["attention_mask"],
            "drug_input_ids":         drug_inputs["input_ids"],
            "drug_attention_mask":    drug_inputs["attention_mask"],
        }

        cosine_sim = model(inputs)["cosine_similarity"]
        pkd_preds  = model.cosine_similarity_to_pkd(
            cosine_sim,
            pkd_upper_bound=pkd_upper_bound,
            pkd_lower_bound=pkd_lower_bound,
        )

        predictions.extend(pkd_preds.cpu().tolist())
        labels.extend(batch_ys)

    return predictions, labels


def compute_and_print_metrics(labels, predictions, dataset_name):
    t_labels = torch.tensor(labels)
    t_preds  = torch.tensor(predictions)

    rmse     = get_rmse(t_labels, t_preds)
    pearson  = get_pearson(t_labels, t_preds)
    spearman = get_spearman(t_labels, t_preds)
    ci       = get_ci(t_labels, t_preds)

    print(f"\n=== Results on {dataset_name} ===")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  CI:       {ci:.4f}")

    return {
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "ci": float(ci),
    }


def save_results(df, predictions, labels, metrics, output_csv, dataset_name):
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    out_df = df[["Drug", "Target", "Y"]].copy()
    out_df["prediction"] = predictions
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    metrics_path = output_csv.replace(".csv", "_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    fig, ax = plt.subplots()
    sns.regplot(x=labels, y=predictions, ax=ax)
    ax.set_title(f"Fine-tuned BALM on {dataset_name}")
    ax.set_xlabel("Experimental pKd (or KIBA score)")
    ax.set_ylabel("Predicted pKd")
    plot_path = output_csv.replace(".csv", "_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--train_csv",        type=str, required=True)
    parser.add_argument("--test_csv",         type=str, required=True)
    parser.add_argument("--dataset",          type=str, required=True, choices=["davis", "kiba"])
    parser.add_argument("--output_csv",       type=str, default="results/predictions.csv")
    parser.add_argument("--device",           type=str, default="cuda:0")
    parser.add_argument("--epochs",           type=int, default=5)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument(
        "--bindingdb_csv", type=str, default=None,
        help="Optional: path to BindingDB data.csv to remove overlapping pairs from train and test sets."
    )
    parser.add_argument(
        "--save_model", type=str, default=None,
        help="Optional: path to save fine-tuned model weights, e.g. results/davis_finetuned_model.bin"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Optional: path to a previously saved .bin file to resume fine-tuning from, e.g. results/davis_finetuned_model.bin"
    )
    return parser.parse_args()


def main():
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    lr           = configs.model_configs.model_hyperparameters.learning_rate
    protein_max  = configs.model_configs.model_hyperparameters.protein_max_seq_len
    drug_max     = configs.model_configs.model_hyperparameters.drug_max_seq_len


    print("Loading pretrained model...")


    model = BALM(configs.model_configs)
    model = load_trained_model(model, configs.model_configs, is_training=True)
    model = model.to(args.device)

    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        resume_state = torch.load(args.resume_from, map_location=args.device)
        model.load_state_dict(resume_state)
        del resume_state          # free the temporary state dict copy
        torch.cuda.empty_cache()  # release GPU memory back to pool
        print("Checkpoint loaded successfully.")

    protein_tokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.protein_model_name_or_path
    )
    drug_tokenizer = AutoTokenizer.from_pretrained(
        configs.model_configs.drug_model_name_or_path
    )


    train_df = load_csv(args.train_csv)
    test_df  = load_csv(args.test_csv)

    # Remove BindingDB overlaps from test set only — keeps evaluation honest
    # (training on BindingDB-overlapping pairs is fine)
    if args.bindingdb_csv:
        print("\nChecking for BindingDB overlap in test set...")
        bdb = pd.read_csv(args.bindingdb_csv, usecols=["Drug", "Target"])
        bdb_pairs = set(zip(bdb["Drug"], bdb["Target"]))

        before = len(test_df)
        test_df = test_df[test_df.apply(lambda r: (r["Drug"], r["Target"]) not in bdb_pairs, axis=1)].reset_index(drop=True)
        print(f"  Test: removed {before - len(test_df)} overlapping pairs ({len(test_df)} remaining)")

    # Compute pKd bounds from training data — correct for both Davis and KIBA
    # (KIBA is not pKd scale so using BindingDB checkpoint bounds would be wrong)
    pkd_lower_bound = float(train_df["Y"].min())
    pkd_upper_bound = float(train_df["Y"].max())
    print(f"pKd bounds from {args.dataset} training data: [{pkd_lower_bound:.4f}, {pkd_upper_bound:.4f}]")


    print(f"\nFine-tuning for {args.epochs} epochs on {len(train_df)} training samples...")
    model = finetune(
        model=model,
        train_df=train_df,
        test_df=test_df,
        protein_tokenizer=protein_tokenizer,
        drug_tokenizer=drug_tokenizer,
        pkd_lower_bound=pkd_lower_bound,
        pkd_upper_bound=pkd_upper_bound,
        device=args.device,
        epochs=args.epochs,
        learning_rate=lr,
        batch_size=args.batch_size,
        protein_max_len=protein_max,
        drug_max_len=drug_max,
        save_model_path=args.save_model,
        eval_every=5,
    )


    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"\nFine-tuned model saved to {args.save_model}")


    print(f"\nRunning inference on {len(test_df)} test samples...")
    predictions, labels = run_inference(
        model=model,
        df=test_df,
        protein_tokenizer=protein_tokenizer,
        drug_tokenizer=drug_tokenizer,
        pkd_lower_bound=pkd_lower_bound,
        pkd_upper_bound=pkd_upper_bound,
        device=args.device,
        batch_size=args.batch_size,
        protein_max_len=protein_max,
        drug_max_len=drug_max,
    )

    metrics = compute_and_print_metrics(labels, predictions, args.dataset)
    save_results(test_df, predictions, labels, metrics, args.output_csv, args.dataset)


if __name__ == "__main__":
    main()

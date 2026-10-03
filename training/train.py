"""
Training script for Swara models (Production-Ready Infrastructure).
Executes training loop, handles checkpointing, early stopping, and exports saved models.

IMPORTANT:
- Requires real training dataset in data/raw/ (silence, unknown, swara).
- Recording-level train/val/test splits (NOT speaker-independent).
- Never uses fake/random/zero data.
- Refuses to train if any required class is absent.
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure training directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set deterministic random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    tf = None

from model import build_dscnn_model
from dataset import SwaraDataset
from config import CLASSES, CLASS_TO_IDX, DEFAULT_NUM_FILTERS, DEFAULT_NUM_CLASSES


def train(
    data_dir: str = "data/raw",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = "models/swara_saved_model",
    num_filters: int = DEFAULT_NUM_FILTERS,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dry_run: bool = False,
):
    """
    Execute training pipeline once real training data is present.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for model training. Please install tensorflow.")

    print("=" * 65)
    print("        Swara Wake Word Model Training Pipeline (M2c.1)        ")
    print("=" * 65)
    print(f"Data directory:       {os.path.abspath(data_dir)}")
    print(f"Model architecture:   DS-CNN ({num_filters} filters, {DEFAULT_NUM_CLASSES} classes)")
    print(f"Batch size:           {batch_size}")
    print(f"Epochs:               {epochs}")
    print(f"Learning rate:        {learning_rate}")
    print(f"Export target:        {save_path}")
    print(f"Deterministic seed:   {RANDOM_SEED}")
    print("Partitioning:         RECORDING-LEVEL (NOT speaker-independent)")
    print("=" * 65)

    dataset_loader = SwaraDataset(data_dir=data_dir)
    splits = dataset_loader.scan_dataset()
    total_files = dataset_loader.count_total_files(splits)

    if total_files == 0:
        raise FileNotFoundError(
            f"\n[ERROR] No valid WAV audio files found in '{os.path.abspath(data_dir)}'.\n"
            "Real training data is required to train Swara.\n"
            "Expected structure:\n"
            f"  {data_dir}/silence/*.wav\n"
            f"  {data_dir}/unknown/*.wav\n"
            f"  {data_dir}/swara/*.wav\n"
            "Do NOT use fake, random, or synthetic placeholder audio for training."
        )

    # Validate that all required classes are present
    classes_present = dataset_loader.get_classes_present(splits)
    missing_indices = set(range(len(CLASSES))) - classes_present
    if missing_indices:
        missing_names = [CLASSES[i] for i in missing_indices]
        raise ValueError(
            f"\n[ERROR] Required audio classes missing from '{data_dir}': {missing_names}.\n"
            "All 3 classes ('silence', 'unknown', 'swara') must contain real audio files before training can begin."
        )

    print(f"Dataset scanned: {total_files} total audio recordings.")
    print(f"  * Train set: {len(splits['train'])} recordings")
    print(f"  * Val set:   {len(splits['val'])} recordings")
    print(f"  * Test set:  {len(splits['test'])} recordings")

    # Load audio features
    print("\nExtracting 49x10 MFCC features for train and val splits...")
    X_train, y_train = dataset_loader.load_tensors_from_file_list(splits["train"], augment_swara=True)
    X_val, y_val = dataset_loader.load_tensors_from_file_list(splits["val"], augment_swara=False)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

    # Build model
    model = build_dscnn_model(
        input_shape=(49, 10, 1),
        num_classes=len(CLASSES),
        num_filters=num_filters,
    )
    if model is None:
        raise RuntimeError("Failed to build DS-CNN model.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    if dry_run:
        print("\n[DRY RUN] Verification successful. Skipping actual model.fit execution.")
        return model, None

    # Checkpoint and history logging
    save_dir = os.path.dirname(save_path) if (save_path.endswith(".keras") or "." in os.path.basename(save_path)) else save_path
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(save_dir, "best_val_model.keras")

    # Balanced class weights to handle dataset class imbalance
    class_counts = {c: int(np.sum(y_train == c)) for c in range(len(CLASSES))}
    total_train = len(y_train)
    class_weights = {
        c: float(total_train) / (len(CLASSES) * max(1, count))
        for c, count in class_counts.items()
    }
    print(f"Class distribution in train split: {class_counts}")
    print(f"Computed balanced class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(save_dir, "training_history.csv"),
            separator=",",
            append=False,
        ),
    ]

    print(f"\nStarting model training for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
    )

    if os.path.exists(checkpoint_filepath):
        try:
            best_model = tf.keras.models.load_model(checkpoint_filepath)
            val_loss_current, val_acc_current = model.evaluate(X_val, y_val, verbose=0)
            val_loss_best, val_acc_best = best_model.evaluate(X_val, y_val, verbose=0)
            print(f"Final epoch val_loss: {val_loss_current:.4f} (acc: {val_acc_current*100:.2f}%)")
            print(f"Best checkpoint val_loss: {val_loss_best:.4f} (acc: {val_acc_best*100:.2f}%)")
            if val_loss_best <= val_loss_current:
                print(f"Selecting best checkpoint (val_loss: {val_loss_best:.4f}, val_accuracy: {val_acc_best*100:.2f}%) for export.")
                model = best_model
        except Exception as e:
            print(f"[Note] Checkpoint selection note: {e}")

    keras_save_path = save_path if save_path.endswith(".keras") else f"{save_path}.keras"
    print(f"\nTraining completed. Exporting final model to {keras_save_path}...")
    model.save(keras_save_path)
    try:
        if not save_path.endswith(".keras"):
            model.export(save_path)
            print(f"Exported SavedModel to {save_path}")
    except Exception as e:
        print(f"[Note] SavedModel bundle export note: {e}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swara DS-CNN Wake Word Model")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--save_path", type=str, default="models/swara_saved_model", help="Path to export trained model")
    parser.add_argument("--num_filters", type=int, default=DEFAULT_NUM_FILTERS, help="DS-CNN channel count")
    parser.add_argument("--dry_run", action="store_true", help="Perform setup & data loading check without invoking fit")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        num_filters=args.num_filters,
        dry_run=args.dry_run,
    )

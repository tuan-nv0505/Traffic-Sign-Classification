import os
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
import torch
from typing import Type
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from datasets.gtsrb_dataset import GTSRBDataset
from models.super_mamba import SuperMamba
from utils import get_args, plot_confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

args = get_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.lr
DEVICE = torch.device(args.device)
TRAIN_DATA = args.train_data
FOLDS = args.folds
WORKERS = args.workers
TRAINED = args.trained
LOGGING = args.logging
LOAD_CHECKPOINT = args.load_checkpoint
TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
])


def train(Dataset: Type[GTSRBDataset]):
    train_dataset = Dataset(path=TRAIN_DATA, transforms=TRANSFORMS)
    indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.labels)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    folds = [(train_idx, validation_idx) for train_idx, validation_idx in skf.split(indices, labels)]

    start_fold = 0
    if os.path.exists(TRAINED):
        trained_folds = [
            int(d.replace("fold_", ""))
            for d in os.listdir(TRAINED)
            if d.startswith("fold_") and os.path.isdir(os.path.join(TRAINED, d))
        ]
        if len(trained_folds) > 0:
            start_fold = max(trained_folds) - 1

    for fold in range(start_fold, FOLDS):
        print(f"\n" + "=" * 20 + f" TRAINING FOLD {fold + 1}/{FOLDS} " + "=" * 20)

        train_idx, validation_idx = folds[fold]
        fold_trained_path = os.path.join(TRAINED, f"fold_{fold + 1}")
        fold_logging_path = os.path.join(LOGGING, f"fold_{fold + 1}")
        os.makedirs(fold_trained_path, exist_ok=True)
        os.makedirs(fold_logging_path, exist_ok=True)

        train_dataloader = DataLoader(
            dataset=Subset(train_dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=False
        )

        validation_dataloader = DataLoader(
            dataset=Subset(train_dataset, validation_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            drop_last=False
        )

        model = SuperMamba(dims=3, depth=4, num_classes=43).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_path = os.path.join(fold_trained_path, "checkpoint.pth")
        best_checkpoint_path = os.path.join(fold_trained_path, "best_checkpoint.pth")

        start_epoch = 0
        best_accuracy = 0

        if os.path.exists(checkpoint_path) and LOAD_CHECKPOINT:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint["epoch"]
                    best_accuracy = checkpoint.get("best_accuracy", 0)
                    print(f"Resume Fold {fold + 1} from epoch {start_epoch + 1}")
                except Exception as ex:
                    print(f"Load checkpoint failed for fold {fold + 1}: {ex}")
        else:
            print(f"Start training for Fold {fold + 1}")

        num_iterations = len(train_dataloader)
        writer = SummaryWriter(fold_logging_path)

        for epoch in range(start_epoch, EPOCHS):
            progress_bar = tqdm(train_dataloader)
            model.train()
            total_loss = 0.0
            for i, (images, labels_batch) in enumerate(progress_bar):
                images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)

                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels_batch)
                loss.backward()
                optimizer.step()

                progress_bar.set_description(
                    f"Fold [{fold + 1}/{FOLDS}] - "
                    f"Epoch [{epoch + 1}/{EPOCHS}] - "
                    f"Loss: {loss.item():.4f}"
                )
                total_loss += loss.item()
                writer.add_scalar(f"Fold {fold + 1}/Train/Loss", loss.item(), epoch * num_iterations + i)
            print(f"--> Loss for epoch {epoch + 1:.4f} : {total_loss / num_iterations}")

            model.eval()
            list_prediction = []
            list_label = []
            for images, labels_val in validation_dataloader:
                images = images.to(DEVICE)
                labels_val = labels_val.to(DEVICE)

                with torch.no_grad():
                    outputs = model(images)
                list_prediction.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                list_label.extend(labels_val.cpu().numpy())

            accuracy = accuracy_score(list_label, list_prediction)
            print(f"Fold {fold + 1} - Epoch {epoch + 1} Accuracy: {accuracy:.4f}")

            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
            }

            torch.save(checkpoint, checkpoint_path)
            if is_best:
                torch.save(checkpoint, best_checkpoint_path)
                print(f"--> New Best Accuracy for Fold {fold + 1}")

            writer.add_scalar(f"Fold {fold + 1}/Validation/Accuracy", accuracy, epoch + 1)
            plot_confusion_matrix(
                writer=writer,
                cm=confusion_matrix(list_label, list_prediction),
                class_names=train_dataset.categories,
                epoch=epoch + 1,
                mode="precision",
            )

            plot_confusion_matrix(
                writer=writer,
                cm=confusion_matrix(list_label, list_prediction),
                class_names=train_dataset.categories,
                epoch=epoch + 1,
                mode="recall",
            )

        writer.close()


if __name__ == '__main__':
    train(GTSRBDataset)
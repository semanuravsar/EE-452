from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json



def create_graph_loaders(graph_dataset, val_ratio=0.2, batch_size=32, shuffle=True):
    val_size = int(len(graph_dataset) * val_ratio)
    train_size = len(graph_dataset) - val_size

    train_dataset, val_dataset = random_split(graph_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits.view(-1), batch.y.view(-1).float())
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).view(-1)
        preds = probs > 0.5
        targets = batch.y.view(-1).bool()

        total_loss += loss.item() * batch.num_graphs
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / len(loader.dataset), acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits.view(-1), batch.y.view(-1).float())

            probs = torch.sigmoid(logits).view(-1)
            preds = probs > 0.5
            targets = batch.y.view(-1).bool()

            total_loss += loss.item() * batch.num_graphs
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())

    acc = correct / total
    f1 = f1_score(y_true, [p > 0.5 for p in y_pred], average='macro')
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = float('nan')
    return total_loss / len(loader.dataset), acc, f1, auroc


def train_model(model, train_loader, val_loader, device, lr=2e-4, epochs=1000, patience=150, save_path="best_model.pt"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('update')
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=50)

    best_val_f1 = 0.0
    early_stop_counter = 0


    # Store metrics across epochs
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auroc": []
    }

    # saving path for metrics
    metrics_path = save_path.replace(".pt", "_metrics.json")

    # saving path for history
    history_path = save_path.replace(".pt", "_training_history.json")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auroc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        # Store in history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auroc"].append(val_auroc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUROC: {val_auroc:.4f}")

        

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"*** Saved new best model (Val F1: {val_f1:.4f})")

            
            with open(metrics_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_auroc": val_auroc
                }, f, indent=2)

            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("--- Early stopping triggered")
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm


SEQ_LEN = 256  # nº de tokens de entrada do modelo
D_MODEL = 64  # nº de dimensões de embedding
N_HEADS = 4  # nº de cabeças utilizadas no multi-head attention
Nx = 2  # nº de vezes que é repassado no multi-head attention
N_OUTPUT = 5  # nº de classes de saida

LR = 1e-5  # Learning Rate
BATCH_SIZE = 32  # Batch Size
EPOCHS = 10  # épocas de trainamento

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def eval_model(model, data_eval) -> float:
    model.eval()
    accuracy_metric = Accuracy(task="multiclass", num_classes=N_OUTPUT).to(device)
    
    for batch in tqdm(data_eval):
        labels, tokens = batch['label'].to(device), batch['tokens'].to(device)
        outputs = model(tokens)
        predictions = torch.argmax(outputs, dim=1)
        accuracy_metric.update(predictions, labels)
    
    accuracy = accuracy_metric.compute().item()
    return round(accuracy, 3)

def train_model(
    model,
    data_train,
    data_eval,
    epochs,
    optimizer,
    criterion,
    scheduler=None,
    save_path="models/yepreview_model_.pth",
    early_stopping_patience=2
) -> None:
    accuracy_metric = Accuracy(task="multiclass", num_classes=N_OUTPUT).to(device)
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        accuracy_metric.reset()
        
        batch_iterator = tqdm(data_train, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in batch_iterator:
            labels, tokens = batch['label'].to(device), batch['tokens'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy_metric.update(predictions, labels)

            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{total_loss / (batch_iterator.n + 1):.4f}",
                "accuracy": f"{accuracy_metric.compute().item():.4f}"
            })
        
        # Evaluate at the end of the epoch
        eval_accuracy = eval_model(model, data_eval)
        # Check for improvement
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            epochs_no_improve = 0
            
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_accuracy:.3f}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break
        print(f"Validation Accuracy after epoch {epoch + 1}: {eval_accuracy:.3f}")
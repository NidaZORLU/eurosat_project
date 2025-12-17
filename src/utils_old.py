import torch
from tqdm import tqdm


def accuracy_from_logits(logits, targets):
    """
    Burda logitsten accuracy hesaplarız.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    1 epoch boyunca modeli eğitir.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy_from_logits(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device, desc="Val"):
    """
    Modeli validation veya test setinde değerlendirir.
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_acc += accuracy_from_logits(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_acc / len(dataloader.dataset)

    return epoch_loss, epoch_acc

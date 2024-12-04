import torch
from tqdm import tqdm


def evaluate_model(model, test_loader, device="gpu" if torch.cuda.is_available() else "cpu"):
    model.eval()  # turn train mode OFF

    total = 0
    correct = 0

    with torch.no_grad():  # No need to compute gradients
        eval_bar = tqdm(test_loader, desc="Evaluating")
        for images, labels in eval_bar:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_acc = 100 * correct / total
            eval_bar.set_postfix(current_acc=f"{current_acc:.2f}%")

    print(f"Final Accuracy: {100 * correct / total:.2f}%")
    
    
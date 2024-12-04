import torch
import numpy as np
import matplotlib.pyplot as plt



def imshow(img, cmap='gray'):
    npimg = img[0].numpy() if isinstance(img, tuple) else img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    plt.show()

def display_model_predictions(model, test_loader, num_images=5, cmap='gray', classes_dict=None):
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    if classes_dict is not None:
        labels = [classes_dict[l.item()] for l in labels]
        predicted = [classes_dict[p.item()] for p in predicted]
    
    rows = int(np.ceil(num_images / 5))
    columns = 5
    _, axes = plt.subplots(rows, columns, figsize=(12, 6))
    
    for i in range(rows * columns):
        if i < num_images:
            current_row = i // 5
            image = images[i] / 2 + 0.5  # Unnormalize
            if rows > 1:
                axes[current_row, i % 5].imshow(image.permute(1, 2, 0).numpy(), cmap=cmap)
                axes[current_row, i % 5].set_title(f"Label: {labels[i]},\nPredicted: {predicted[i]}")
                axes[current_row, i % 5].axis("off")
            else:
                axes[i].imshow(image.permute(1, 2, 0).numpy(), cmap=cmap)
                axes[i].set_title(f"Label: {labels[i]},\nPredicted: {predicted[i]}")
                axes[i].axis("off")
        else:
            if rows > 1:
                axes[current_row, i % 5].axis("off")
            else:
                axes[i].axis("off")
    plt.tight_layout()
    plt.show()
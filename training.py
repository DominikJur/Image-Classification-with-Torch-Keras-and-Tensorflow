import time
import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, num_epochs=5, device="gpu" if torch.cuda.is_available() else "cpu"):
    model.train()  
    
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # Start timing epoch
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  
                progress_bar.set_postfix(current_loss=loss.item())
                running_loss = 0.0
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time // 60:.0f} minutes and {epoch_time % 60:.0f} seconds")
        
        # Optionally save the model checkpoint
        torch.save(model.state_dict(), f"{type(model).__name__}_epoch_{epoch+1}.pth")

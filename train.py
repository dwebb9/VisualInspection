import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torchvision import datasets, transforms

from network import CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train():
    ############    Set all params    ############
    epochs = 20
    batch_size = 42
    img_size = 64 # TODO : Figure out what this really should be
    train_size = 0.8

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using ", device)

    # Load all data in
    dataset = datasets.ImageFolder(root='./data',
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
                           
    # split data
    n = len(dataset)
    n_test = int(train_size * n)
    data_train = Subset(dataset, range(n_test))
    data_val = Subset(dataset, range(n_test, n))
    # Make loaders
    train_loader = DataLoader(data_train, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=batch_size)
    # TODO : setup test/train split
    
    # init model
    model = CNN(dataset).to(device)

    # init optimizer and other things
    objective = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # set up places to save things
    train_accur = []
    temp_train_accur = []
    val_accur = []
    num_batch = len(train_loader)
    loop = tqdm(total=num_batch*epochs, position=0)

    # do the training!
    for epoch in range(epochs):
        #iterate through batches
        for batch, (x, y_truth) in enumerate(train_loader):
            x, y_truth = x.to(device), y_truth.to(device)
            
            opt.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            opt.step()
            
            temp_train_accur.append((y_hat.argmax(1) == y_truth).float().mean().item())

            loop.set_description(f"accur: {temp_train_accur[-1]}, loss: {loss.item()}")
            loop.update(1)
                    
            #check validation loss every 25 epochs
            if batch % 400 == 0:
                train_accur.append( (epoch+batch/num_batch, np.mean(temp_train_accur)) )
                temp_train_accur = []
                
                temp = [ (model(x.to(device)).argmax(1) == y.to(device)).float().mean().item() for x, y in val_loader]
                val_accur.append( (epoch+batch/num_batch, np.mean(temp)) )

    # plot results
    plt.figure(figsize=(10,6))
    l1, l2 = zip(*train_accur)
    plt.plot(l1, l2, label="Training Accuracy")
    l1, l2 = zip(*val_accur)
    plt.plot(l1, l2, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # save model
    torch.save('model.pkl', model.state_dict())

if __name__ == "__main__":
    train()
from Siam import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    dataset = TrackingDataset('data/videos', 'data/annotations', transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = SiameNN().to(device)
    opt = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for z, x, label_map in loader:
            z, x, label_map = z.to(device), x.to(device)
            fz = model(z)
            fx = model(x)
            out = cross_relation(fz, fx)
            loss = torch.mean(-label_map * torch.log(out + 1e-10) - (1 - label_map) * torch.log(1 - out + 1e-10))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss={total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), 'sia.pth')

if __name__ == '__main__':
    train()
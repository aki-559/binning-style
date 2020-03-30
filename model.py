import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def train(model, device, loader, optimizer, epoch):
    model.train()
    data_size = len(loader)
    
    start = time.time()
    for batch_idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y =  y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        print("\rTrain Epoch: {} [ {:0=5}/{:0=5} ({:.0f}%)]\t Loss: {:.4f}".format(epoch,
            batch_idx+1, data_size, (batch_idx + 1) * 100. / data_size, loss.item()
        ), end="")

    end = time.time()

    print("\tThis took {:.3f} seconds".format(end-start), end="")

def test(model, device, loader):
    model.eval()

    val_loss = 0
    true = 0
    data_size = len(loader) * loader.batch_size

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)

            val_loss += F.nll_loss(out, y, reduction="sum").item()
            y_pred = out.argmax(dim=1, keepdim=True)
            true += y_pred.eq(y.view_as(y_pred)).sum().item()

    val_loss /= data_size
    true /= data_size

    print("\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%".format(val_loss, true*100))
    
    return val_loss

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layers = nn.Sequential(
            self.conv_relu(4, 128, 3, 1),
            self.conv_relu(128, 128, 3, 1),
            self.conv_relu(128, 256, 3, 1),
            self.conv_relu(256, 256, 3, 1),
            self.conv_relu(256, 512, 3, 1),
            self.conv_relu(512, 512, 3, 1),
            self.conv_relu(512, 1024, 3, 1),
            self.conv_relu(1024, 1024, 3, 1)
        )

        self.fc = nn.Linear(1024, output_dim)

        self.n_module = 8

    def forward(self, x):
        # module * 5
        x = self.conv_layers(x)

        # gap
        x = torch.mean(x, dim=2)

        # fc
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    def conv_relu(self, input_dim, output_dim, kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

    def get_feature_map(self, x, layer):
        x = self.conv_layers[:layer](x)

        return x

    def get_gap(self,x):
        x = self.conv_layers(x)
        x = torch.mean(x, dim=2)

        return x

    def get_style(self, x, layer):
        x = self.conv_layers[:layer](x)
        x = torch.squeeze(x)
        gram = torch.mm(x, torch.t(x))
        
        return gram

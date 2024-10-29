import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class CustomLoss(nn.Module):
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        assert outputs.shape[1] == 3, "Outputs and weights must be for 3 elements"
        component_losses = []
        total_loss = 0
        for i in range(3):
            loss = (outputs[:, i] - targets[:, i]) ** 2
            weighted_loss = self.weights[i] * loss.mean()
            component_losses.append(weighted_loss.item())
            total_loss += weighted_loss
        return total_loss, component_losses


def train_ensemble_with_bagging(X_train, y_train, X_val, y_val, num_models=5, num_epochs=2000, hidden_dim=32, lr=0.001, weights=[1, 1, 1], patience=100):
    models = []
    for i in range(num_models):
        bootstrap_indices = torch.randint(0, X_train.shape[0], (X_train.shape[0],))
        X_train_bootstrap = X_train[bootstrap_indices]
        y_train_bootstrap = y_train[bootstrap_indices]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = FNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=y_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = CustomLoss(weights)

        best_loss = float('inf')
        epochs_no_improve = 0
        for epoch in tqdm(range(num_epochs), desc=f"Training Model {i + 1}/{num_models}", unit="epoch"):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_bootstrap)
            loss, _ = loss_fn(outputs, y_train_bootstrap)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss, _ = loss_fn(val_outputs, y_val)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'results/best_model_ensemble_{i}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f'Early stopping triggered for model {i + 1} after {epoch + 1} epochs.')
                    break

        model.load_state_dict(torch.load(f'results/best_model_ensemble_{i}.pth'))
        models.append(model)

    return models

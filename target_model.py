import torch
import random
from initializations import activation_ids
from sklearn.metrics import accuracy_score
# L -> number of layers
# N -> in_features for whole network
# M -> out_features for whole network
# f_in -> in_features for a particular layer
# f_out -> out_features for a particular layer


@torch.no_grad()
def evaluate(model, data, num_steps):
    model.eval()
    total_loss = 0.
    metrics = []
    for step, (x, y) in enumerate(data):
        if step >= num_steps:
            break
        x = x.reshape(x.shape[0], -1)
        y_pred = model(x.cuda())
        loss = torch.nn.functional.nll_loss(y_pred, y.cuda())
        metrics.append(accuracy_score(y, y_pred.argmax(1).cpu()))
        # total_loss += loss.item()
    # return total_loss / num_steps
    return sum(metrics) / num_steps


def train_target(model, train, val, test, num_steps):
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    total_loss_train = 0.
    train_metrics = []
    for step, (x, y) in enumerate(train):
        if step >= num_steps:
            break
        opt.zero_grad()

        x = x.reshape(x.shape[0], -1)
        y_pred = model(x.cuda())
        # print(y_pred)
        loss = torch.nn.functional.cross_entropy(y_pred, y.cuda())
        train_metrics.append(accuracy_score(
            y, y_pred.argmax(1).detach().cpu()))
        loss.backward()
        opt.step()

        total_loss_train += loss.item()
    train_metric = sum(train_metrics) / num_steps
    val_metric = evaluate(model, val, num_steps)
    test_metric = evaluate(model, test, num_steps)
    # total_loss_train = total_loss_train / num_steps

    return train_metric, val_metric, test_metric


def generate_model(in_features, out_features, num_layers):
    layers = []
    f_in = in_features
    activations = list(activation_ids.keys())
    for i in range(num_layers - 1):
        # print(f_in, i, (f_in - (num_layers - i - 1) * out_features + 1))
        f_out = random.randint(
            f_in, f_in + (num_layers - i + 1) * out_features + 1)
        activation = random.choice(activations)
        layers.extend([torch.nn.Linear(f_in, f_out), activation()])
        f_in = f_out

    activation = torch.nn.Identity  # random.choice(activations)
    layers.extend([torch.nn.Linear(f_in, out_features), activation(dim=1)])
    return torch.nn.Sequential(*layers)


if __name__ == "__main__":
    inp = torch.randn(1, 768)
    i = 0
    while True:
        num_layers = random.randint(2, 10)
        print(i, num_layers)
        model = generate_model(768, 10, num_layers)
        model(inp)
        i += 1
    # print(model)

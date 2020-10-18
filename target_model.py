import torch
import random
from initializations import activation_ids
# L -> number of layers
# N -> in_features for whole network
# M -> out_features for whole network
# f_in -> in_features for a particular layer
# f_out -> out_features for a particular layer

def generate_model(in_features, out_features, num_layers):
    layers = []
    f_in = in_features
    activations = list(activation_ids.keys())
    for i in range(num_layers - 1):
        # print(f_in, i, (f_in - (num_layers - i - 1) * out_features + 1))
        f_out = random.randint(f_in, f_in + (num_layers - i + 1) * out_features + 1)
        activation = random.choice(activations)
        layers.extend([torch.nn.Linear(f_in, f_out), activation()])
        f_in = f_out
    
    activation = random.choice(activations)    
    layers.extend([torch.nn.Linear(f_in, out_features), activation()])
    return torch.nn.Sequential(*layers)

if __name__ == "__main__":
    
    i = 0
    while True:
        num_layers = random.randint(2, 10)
        print(i, num_layers)
        model = generate_model(768, 10, num_layers)
        i += 1
    # print(model)
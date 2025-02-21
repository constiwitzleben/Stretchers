import torch.nn as nn
import torch

class Embedded_Conditional_Residual_MLP(nn.Module):
    def __init__(self, input_dim, parameter_dim, output_dim, hidden_dim=256, embed_dim=64, num_layers = 1):
        super(Embedded_Conditional_Residual_MLP, self).__init__()

        self.parameter_embed = nn.Sequential(
            nn.Linear(parameter_dim, embed_dim)
        )
        
        layers = []

        if num_layers == 0:
            layers.append(nn.Linear(input_dim + embed_dim, output_dim))
        else:
            # First hidden layer
            layers.append(nn.Linear(input_dim + embed_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, parameters):
        embedded_parameters = self.parameter_embed(parameters)
        combined = torch.cat([x, embedded_parameters], dim = -1)
        residual = self.mlp(combined)
        return x + residual
    
class Embedded_Conditional_Fully_Residual_MLP(nn.Module):
    def __init__(self, input_dim, parameter_dim, output_dim, hidden_dim=256, embed_dim=64, num_layers = 1):
        super(Embedded_Conditional_Fully_Residual_MLP, self).__init__()

        self.parameter_embed = nn.Sequential(
            nn.Linear(parameter_dim, embed_dim)
        )

        self.input_layer = nn.Linear(input_dim + embed_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, x, parameters):

        if torch.all(parameters==0):
            return x
        else:
            embedded_parameters = self.parameter_embed(parameters)
            combined = torch.cat([x, embedded_parameters], dim = -1)
    
            out = self.input_layer(combined)
            out = torch.relu(out)
    
            for layer in self.hidden_layers:
                residual = out
                out = torch.relu(layer(out)) + residual
    
            residual_out = self.output_layer(out)
            
            return x + residual_out
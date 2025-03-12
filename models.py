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
        
class SuperNet(nn.Module):
    def __init__(self, descriptor_dim=256, parameter_dim=3, hidden_dim=256, num_layers=2):
        super(SuperNet, self).__init__()
        
        self.p_scale = nn.Parameter(torch.ones(1))  # Learnable scale parameter
        
        # Create the first fusion layer
        self.input_fc = nn.Linear(descriptor_dim + parameter_dim, hidden_dim)
        
        # Dynamically create a list of hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        
        # Output layer
        self.output_fc = nn.Linear(hidden_dim, descriptor_dim)
        
    def forward(self, x, p):

        if torch.all(p==0):
            return x
        
        # Scale affine parameters to match descriptor magnitude
        scaled_p = p * self.p_scale
        
        # Concatenate descriptor and affine parameters
        combined = torch.cat([scaled_p, x], dim=1)
        out = nn.functional.relu(self.input_fc(combined))
        
        # Pass through hidden layers
        for hidden_layer in self.hidden_layers:
            out = nn.functional.relu(hidden_layer(out))
        
        residual = self.output_fc(out)
        
        return residual
    
class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim,num_layers):
        super(MLP, self).__init__()

        if num_layers == 0:
            self.model = nn.Linear(input_dim,output_dim)
        else:
            layers = []
            layers.append(nn.Linear(input_dim,hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_dim,output_dim))
            self.model = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.model(x)

class TripleNet(nn.Module):
    def __init__(self, descriptor_dim=256, parameter_dim=3, hidden_dim=256, num_layers=2, num_nets=3):
        super(TripleNet, self).__init__()
        
        self.p_scale = nn.Parameter(torch.ones(1))  # Learnable scale parameter
        
        # Create the first fusion layer
        self.mlp_list = nn.ModuleList([MLP(descriptor_dim + parameter_dim, descriptor_dim, hidden_dim, num_layers) 
                                      for _ in range(num_nets)])
        
    def forward(self, x, p):
        
        if torch.all(p==0):
            return x
        
        device = x.device
        
        # Scale affine parameters to match descriptor magnitude
        scaled_p = p * self.p_scale
        
        # Concatenate descriptor and affine parameters
        combined = torch.cat([scaled_p, x], dim=1)
        
        for mlp in self.mlp_list:
            mlp = mlp.to(device)
            x += mlp(combined)
        
        return x
"""The Stretcher descriptor transformation network (paper, Sec. 2.2.2).

Only TripleNet is used. Three earlier ablation architectures
(Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP and
SuperNet) were removed once TripleNet was selected; they remain in the git
history if needed.
"""

import torch
import torch.nn as nn


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
    """The Stretcher descriptor transformation network (paper, Sec. 2.2.2, Eq. 1).

    Three lightweight MLPs, one per strain component, whose outputs are summed and
    added to the original descriptor:  rho(alpha, d) = d + sum_i MLP_i(alpha, d).

    The residual form keeps descriptors stable under small deformations while
    adapting them selectively under larger strain. A zero strain vector is returned
    unchanged, short-circuiting the identity hypothesis.
    """

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
        
        # Out-of-place on purpose. `x += ...` mutates the caller's descriptor
        # tensor, and stretch_descriptions() feeds the same tensor through this
        # network once per deformation hypothesis. On CUDA/MPS the preceding
        # .to(device) silently makes a copy and hides the aliasing, but on CPU
        # .to() is a no-op and every hypothesis after the first compounds the
        # previous one's residual.
        out = x
        for mlp in self.mlp_list:
            out = out + mlp.to(device)(combined)

        return out
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DeDoDe.utils import to_pixel_coords, to_normalized_coords
from .affine_transformations import generate_strain_tensors, generate_27_strain_tensors, generate_larger_strain_tensors

def dual_softmax_matcher(stretched_descriptors: tuple['T','N','D'], base_descriptor: tuple['1','M','D'], inv_temperature = 1, normalize = False, stretch_type = 'normal', verbose = False):

    T, N, D = stretched_descriptors.shape
    base_descriptor = base_descriptor.repeat(T, 1, 1)
    M = base_descriptor.shape[1]

    if normalize:
        stretched_descriptors = stretched_descriptors/stretched_descriptors.norm(dim=-1,keepdim=True)
        base_descriptor = base_descriptor/base_descriptor.norm(dim=-1,keepdim=True)
    
    # corr = torch.einsum("t n d, t m d -> t n m", stretched_descriptors, base_descriptor) * inv_temperature

    corr = torch.einsum("n d, m d -> n m", stretched_descriptors[0], base_descriptor[0]) * inv_temperature
    corr_indices = torch.zeros_like(corr)

    for i in range(1,T):
        corr_next = torch.einsum("n d, m d -> n m", stretched_descriptors[i], base_descriptor[i]) * inv_temperature
        idx = torch.where(corr_next > corr)
        corr = torch.maximum(corr, corr_next)
        corr_indices[idx] = i

    # Chose maximum value over T dimension
    # corr, corr_indices = corr.max(dim = 0)

    corr = corr.unsqueeze(0)
    
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)

    stretch_counts = torch.bincount(corr_indices.flatten().to(torch.int64), minlength=T)

    top5_counts, top5_indices = torch.topk(stretch_counts, 5)
    top5_percents = (top5_counts / stretch_counts.sum()) * 100

    if stretch_type == 'only27':
        tensors = np.array(generate_27_strain_tensors())
    elif stretch_type == 'larger':
        tensors = np.array(generate_larger_strain_tensors())
    else:
        tensors = np.array(generate_strain_tensors())

    if verbose:
        print(f'Top 5 stretches for max similarity:')
        for j in range(5):
            print(f'{tensors[top5_indices[j]]}: {top5_percents[j]:.0f}%')

    return P, corr_indices

class StretcherDualSoftMaxMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0, stretch_type = 'normal', verbose = False):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P, corr_indices = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp, stretch_type = stretch_type,
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]

        matched_transformations = corr_indices[inds[:,1],inds[:,2]]

        stretch_counts = torch.bincount(matched_transformations.to(torch.int64))

        index = 5 if len(stretch_counts) > 5 else len(stretch_counts)

        top5_counts, top5_indices = torch.topk(stretch_counts, index)
        top5_percents = (top5_counts / stretch_counts.sum()) * 100

        if stretch_type == 'only27':
            tensors = np.array(generate_27_strain_tensors())
        elif stretch_type == 'larger':
            tensors = np.array(generate_larger_strain_tensors())
        else:
            tensors = np.array(generate_strain_tensors())

        if verbose:
            print(f'Top 5 stretches for matching:')
            for i in range(index):
                print(f'{tensors[top5_indices[i]]}: {top5_percents[i]:.0f}%')


        # tensors = np.array(generate_strain_tensors())

        # print(f'Top 5 stretches for max similarity:')
        # for transformation, count in zip(unique_transformations, counts):
        #     print(f'{tensors[transformation]}: {count}')

        return matches_A, matches_B, batch_inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
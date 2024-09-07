import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import hyptorch.pmath as pmath

import hyptorch.nn as hypnn

################################################################################


def repulsion_loss(d, r, k, clip_v = 0.01):

    d = 2*r - d

    d = torch.clamp(d, clip_v, 2*r)

    c_k = (2*r)**(k+1) / k

    r = torch.tensor(r, requires_grad=False)

    res = (1/(d**k) - 1/((2*r)**k)) * c_k

    return res


def boundary_loss(args, f1, radius, r_per_point):
    
    normed_f1 = torch.norm(f1, dim=1)

    if args.use_hyperbolic_dist:

        loss = F.relu(normed_f1 - radius + 0.1).mean()

    elif args.use_euclidean_dist:

        dist_to_radius = radius - normed_f1

        loss = repulsion_loss( F.relu(2*r_per_point - dist_to_radius), r_per_point, args.k, 0.01)
      
    return loss.sum()




def hyperbolic_uniform_loss(args, f1, r_per_point):

    hyperbolic_distance_between_two_points = 2*r_per_point

    p_wise_dist = pmath.pairwise_poincare_distances(f1, f1)

    p_wise_dist.fill_diagonal_(1e10)

    #################################################################
    dist_top_negatives = p_wise_dist[p_wise_dist < hyperbolic_distance_between_two_points]

    if args.use_nonlinear_repulsion_loss:

        loss_1 = repulsion_loss(F.relu( hyperbolic_distance_between_two_points -  dist_top_negatives), r_per_point, args.k)

    elif args.use_linear_repulsion_loss:

        loss_1 = F.relu( hyperbolic_distance_between_two_points -  dist_top_negatives)
    
    #################################################################
    loss_mean = loss_1.mean()

    return loss_mean



def augmentation_loss(args, f2, f3):
    
    if args.use_hyperbolic_dist:

        dist_ = pmath.dist(f2, f3).mean()

    elif args.use_euclidean_dist:

        dist_ = torch.sqrt((f2 - f3).pow(2).sum(1) + 1e-5).mean()

    return dist_





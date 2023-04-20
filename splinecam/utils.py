import numpy as np
from scipy.spatial import ConvexHull
import torch

@torch.jit.script
def region_eccentricity_2d(poly,eps=1e-10):
    
    # pairwise distances between vertices
    dist = torch.pdist(poly[:-1])

    max_len = dist.max()
    min_len = dist.min()
    return max_len/(min_len+eps)

@torch.jit.script
def region_area_2d(poly):
    
    poly = poly[:-1]
    
    x,y = poly[:,0],poly[:,1]

    S1 = torch.sum(x*torch.roll(y,-1))
    S2 = torch.sum(y*torch.roll(x,-1))

    area = .5*torch.absolute(S1 - S2)

    return area

# @torch.jit.script
def centrality(polys):
    
    verts = torch.vstack([each[:-1] for each in polys])
    centroid = verts.mean(0)
    
    dist = torch.linalg.norm(verts-centroid,dim=-1)
    
    return dist.mean(0),dist.std(0)

def get_region_statistics(polys):
    
    stats = ['vol_m','vol_std','nverts','nregions','ecc_m','ecc_std','centr_m','centr_std','avg_verts']
    
    centr_m,centr_std = centrality(polys)
    
    vols = torch.stack([region_area_2d(each) for each in polys])
    vol_m = vols.mean()
    vol_std = vols.std()
    
    eccs = torch.stack([region_eccentricity_2d(each) for each in polys])
    ecc_m = eccs.mean()
    ecc_std = eccs.std()
    
    avg_verts = torch.stack(
        [torch.tensor(each.shape[0]-1) for each in polys] # -1 because of first vert being repeated
    ).type(torch.float32).mean() 
    
    return stats,torch.stack([vol_m,vol_std,
                torch.tensor(torch.vstack(polys).shape[0]-len(polys)),
                torch.tensor(len(polys)),
                ecc_m,ecc_std,
                centr_m,centr_std])


def get_sparse_idx(A,idx):
    
    W = []
    for i in idx:
        W.append(A[i])
        
    W = torch.stack(W)
    
    return W

def split_sparse_Ab(A):
    
    A = A.transpose(0,1)
    b = A[-1][...,None]
    
    W = []
    for i in torch.arange(A.shape[0]-1):
        W.append(A[i])
    
    W = torch.stack(W)
    return W.transpose(0,1),b

def create_polytope_2d(scale=1,seed=None,init_points_n=30):
    '''
    Create a random convex V-polytope. Samples random points and takes the convex hull.
    scale: float :: radius scaling
    seed: int :: random seed
    init_points_n int :: number of initial points used to get convex hull
    '''
    rng = np.random.default_rng(seed=seed)
    points = rng.random((init_points_n, 2))   # 30 random points in 2-D
    hull = ConvexHull(points)
    poly = points[hull.vertices]
    poly -= poly.mean()
    poly *= scale
    return np.vstack([poly,poly[:1]])

@torch.jit.script
def verify_collinear(v_new,v1,v2, eps: float = 1e-7):
    '''
    Verify collinearity of point sequence v1,v_new,v2
    '''
    
#     l1 = torch.linalg.norm(v1-v2,axis=-1)
#     l2 = torch.linalg.norm(v_new-v1,axis=-1)
#     l3 = torch.linalg.norm(v_new-v2,axis=-1)
    
    l1 = torch.linalg.norm(v1-v2,dim=-1)
    l2 = torch.linalg.norm(v_new-v1,dim=-1)
    l3 = torch.linalg.norm(v_new-v2,dim=-1)
    
    return torch.allclose(l1,l2+l3,rtol=0.,atol=eps)

## TODO: make this jittable
def get_region_means(regions : list, dims: int, dtype: object = torch.float64, device : str = 'cuda'):
    '''
    finds the means of each region
    '''
    
    n_regions = len(regions)
    means = torch.zeros(n_regions, dims, dtype=dtype, device=device)
    
    for i in range(n_regions):
        means[i] = regions[i].to(device).mean(0)
        
    return means

@torch.jit.script
def get_Abw(q,Wb,incoming_Abw):
    '''
    given activation patterns per region (q), layer weights (Wb) and Abw for each region,
    return the new Abw per region. Note that incoming Abw is non unique but outgoing Abw is unique per region
    '''
    
    activ_Wb = q[...,None] * Wb[None,...] 
    
    Abw = torch.bmm(
        activ_Wb[...,:-1],
        incoming_Abw
    )
    
    Abw[...,-1:] = Abw[...,-1:] + activ_Wb[...,-1:]
    
    return Abw

# @torch.jit.script TODO: Make jittable
def regions_list2vec(regions, repeat_first : bool = True):
    '''
    convert list of cycles to vec and list of lengths
    '''
    regions = regions.copy()
    
    if repeat_first:
        for i in range(len(regions)):
            regions[i] = torch.vstack([
                regions[i],regions[i][:1]
            ])
        
        
    out_cycles = torch.vstack(regions)
    cyc_idx = torch.zeros(out_cycles.shape[0], dtype=torch.int64)
    
    start = 0
    ends = torch.zeros(len(regions),dtype=torch.int64)
    for i in range(len(regions)):
        
        n = regions[i].shape[0]
        cyc_idx[start:start+n] = i
        start += n
        ends[i] = start
        
    return out_cycles,cyc_idx,ends


def split_domain_by_edge(domain):
    """
    Splits a poly/domain up into edge-centroid polys
    """
    centroid = domain[:-1].mean(0)
    
    out_domain = []
    for each1,each2 in zip(domain[:-1],domain[1:]):
        out_domain.append(torch.stack([each1,each2,centroid,each1]))
    
    return out_domain

@torch.jit.script
def get_square_slice_from_one_anchor(anchors,pad_dist=1,seed=None):
    """
    Given one vector as an anchor, takes a randomly oriented slice with the anchor at the center
    """
#     if seed is not None:
#         torch.manual_seed(seed)
    
    assert len(anchors) == 1
    
    centroid = anchors[0]
    
    z1 = torch.randn_like(anchors[0])
    z2 = torch.randn_like(anchors[0])
    
    u1 = z1
    u2 = z2 - (u1.T @ z2)/(u1.T @ u1)*u1

    dirs = torch.vstack([u1,u2])
    dirs /= torch.linalg.norm(dirs,dim=-1,keepdim=True)
    domain = torch.vstack([centroid+pad_dist*dirs,centroid-pad_dist*dirs])
    domain_poly = torch.vstack([domain,domain[:1]])
    
    return domain_poly



@torch.jit.script
def get_square_slice_from_two_anchors(anchors,pad_dist=1,seed=0):
    
    torch.manual_seed(seed)
    
    assert len(anchors) >= 2
    
    centroid = torch.mean(anchors[:2],dim=0)
    
    if len(anchors) < 3:
        u1 = anchors[0]
        z = torch.randn_like(anchors[0])
        u2 = z - (u1.T @ z)/(u1.T @ u1)*u1
        
    else:
        u1 = anchors[0]
        u2 = anchors[2] - (u1.T @ anchors[2])/(u1.T @ u1)*u1
    
    dirs = torch.vstack([u1,u2])
    dirs /= torch.linalg.norm(dirs,dim=-1,keepdim=True)
    domain = torch.vstack([centroid+pad_dist*dirs,centroid-pad_dist*dirs])
    domain_poly = torch.vstack([domain,domain[:1]])
    
    return domain_poly


@torch.jit.script
def get_square_slice_from_centroid(anchors,pad_dist=1,seed=0):
    
    assert len(anchors) == 3
    
    centroid = torch.mean(anchors,dim=0)
    
    u1 = anchors[0]
    u2 = anchors[1] - (u1.T @ anchors[1])/(u1.T @ u1)*u1
    
    dirs = torch.vstack([u1,u2])
    dirs /= torch.linalg.norm(dirs,dim=-1,keepdim=True)
    domain = torch.vstack([centroid+pad_dist*dirs,centroid-pad_dist*dirs])
    domain_poly = torch.vstack([domain,domain[:1]])
    
    return domain_poly

@torch.jit.script
def get_proj_mat(domain):
    
    v1 = domain[1] - domain[0]
    v2 = domain[-2] - domain[0]
    
    v = torch.vstack([v1,v2])
    v /= torch.linalg.norm(v,dim=-1,keepdim=True)
    
    return torch.hstack([v.T,domain.mean(0,keepdim=True).T])


@torch.no_grad()
def get_nneigh_points(dataset,target_classes):
    """
    dataset: torch.dataset
    target_classes: list with 2 classes for which to find nearest neighbors
    """
    dataset = dataset.__dict__
    mask = np.asarray(dataset['targets']) == target_classes[0]
    data1 = torch.from_numpy(
        dataset['data'][mask]
    ).type(torch.float32).cuda().transpose(1,3) #channel first
    data1 = data1.reshape(data1.shape[0],-1)
    
    mask = np.asarray(dataset['targets']) == target_classes[1]
    data2 = torch.from_numpy(
        dataset['data'][mask]
    ).type(torch.float32).cuda().transpose(1,3) #channel first
    data2 = data2.reshape(data2.shape[0],-1)
    
    dist = torch.cdist(data1,data2)
    
    idx1 = torch.argsort(dist.min(-1)[0])
    idx2 = dist[idx1[0]].argsort()

    points = torch.vstack([data1[idx1[0]],data2[idx2[:2]]])
    
    return points.cpu()
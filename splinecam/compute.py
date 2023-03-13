import torch
import numpy as np

import networkx as nx
import igraph as ig
import graph_tool as gt
from graph_tool import topology

from splinecam import utils
from splinecam import graph

import tqdm
import time

@torch.no_grad()
def get_hyp_endpoints(poly,hyps,verify=True):
    
    q = graph.get_intersection_pattern(poly,hyps)
    hyp_v1_v2_idx = graph.edge_hyp_intersections(q.T,poly,hyps)
    poly_lines = graph.make_line_2D(poly[hyp_v1_v2_idx[:,1]],poly[hyp_v1_v2_idx[:,2]])

    poly_int_hyps = hyps[hyp_v1_v2_idx[:,0]]
    v,flag = graph.find_intersection_2D(poly_lines,
                                  poly_int_hyps,
                                  verify=verify)

    v = v.type(poly_lines.type())

    if verify:

        assert flag

        flag = graph.verify_collinear(v,
                                poly[hyp_v1_v2_idx[:,1]],
                                poly[hyp_v1_v2_idx[:,2]]
                                )

        assert flag

    hyp_endpoints = v.reshape(-1,2,v.shape[-1])
    return hyp_endpoints


@torch.no_grad()
def to_next_layer_partition_batched(cycles, Abw, current_layer, NN,
                                    dtype=torch.float64, device='cuda',
                                    batch_size=-1, fwd_batch_size=-1):
    
    if batch_size == -1: ## revert to non-batched
        res_regions, new_cyc_idx = graph.to_next_layer_partition(
            cycles, Abw, current_layer, NN, dtype, device
        )
        return res_regions, new_cyc_idx
    
    vec_cyc,cyc_idx,ends = graph.cycles_list2vec(cycles)
    
#     cycles_next = NN.layers[:current_layer].forward(vec_cyc[:128].to(device))
#     q = NN.layers[current_layer].get_intersection_pattern(cycles_next)
    
#     print(q.shape)
    
    fused_op = lambda x: NN.layers[current_layer].get_intersection_pattern(
        NN.layers[:current_layer].forward(x))
    
        
    q = graph._batched_gpu_op(fused_op, 
                        vec_cyc,
                        workers = 2,
                        out_size=(
                            vec_cyc.shape[0],
                            torch.prod(NN.layers[current_layer].output_shape),
                        ),
                        batch_size = fwd_batch_size, out_device='cpu')                                  
    
    
    n_hyps  = torch.prod(NN.layers[current_layer].output_shape)
    
    ## edge intersections. remove between cycles
    mask = q.T[...,:-1] != q.T[...,1:]
    mask = mask.cpu()
    mask[:,(ends-1)[:-1]] = False
    
    if mask.sum() == 0:
        print('no intersections')
        return 
    
#     del  q
#     del cycles_next
    
    ## get indices for hyps-vertex-cycle triads
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    hyp_vert_cyc_idx = torch.hstack([hyp_vert_idx,cyc_idx[hyp_vert_idx[:,1:]]])
    
    ## assert all cycles occur twice in order
    assert torch.all(hyp_vert_cyc_idx[::2,2] == hyp_vert_cyc_idx[1::2,2])
    
    ## query hyps, only get rows which intersect, create idx map
    inter_hyps_idx = torch.unique(hyp_vert_cyc_idx[:,0])
    print(inter_hyps_idx)
    hyps = NN.layers[current_layer].get_weights(row_idx=inter_hyps_idx).cpu()
    hyp_idx_map = torch.ones(n_hyps,dtype=torch.int64)*(hyps.shape[0]+100) ## initialize with idx out of range
    hyp_idx_map[inter_hyps_idx] = torch.arange(hyps.shape[0], dtype=torch.int64)
    
    ## bring hyps to corresponding cycle inputs
    
    hyps_input = graph._batched_gpu_op_2(
        method = graph.hyp2input,
        data1 = hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]],
        data2 = Abw[hyp_vert_cyc_idx[::2,2]],
        batch_size = batch_size,
        out_size = (hyp_vert_cyc_idx[::2,0].shape[0],1,3),
        dtype = dtype
    )[:,0,:]
    
    uniq_cycle_idx = torch.unique(hyp_vert_cyc_idx[:,-1])
    
###     get intersection with all cycle edges
    endpoints = []
    for target_cycle_idx in tqdm.tqdm(uniq_cycle_idx, desc='Iterating regions'):
        
        vert_mask = cyc_idx==target_cycle_idx
        hyp_mask = hyp_vert_cyc_idx[::2,-1] == target_cycle_idx
        
        endpoints.append(get_hyp_endpoints(
            poly = vec_cyc[vert_mask].to(device),
            hyps = hyps_input[hyp_mask].to(device),
        
        ))
        
    return hyps_input.cpu(),torch.vstack(endpoints).cpu()

def get_partitions_with_db(
    domain,
    T,
    NN,
    fwd_batch_size = 1024,
    batch_size = 128,
    n_workers = 2,
    Abw_batch_size = 16,
):
    
    poly = (T[...,:-1].T @ (domain.T - T[...,-1:])).T
    poly = poly.type(torch.float64)

    start_time = time.time()

    ### Get partitions

    Abw = NN.layers[0].get_weights()[None,...].cpu()

    out_cyc = [poly]


    for current_layer in range(1,len(NN.layers)-1):
        print(f'Current layer {current_layer}')

        out_cyc,out_idx = graph.to_next_layer_partition_batched(
            cycles = out_cyc,
            Abw = Abw,
            NN = NN,
            current_layer = current_layer,
            dtype = torch.float64,
            batch_size=batch_size,
            fwd_batch_size=fwd_batch_size,
        )

        with torch.no_grad():

            means = utils.get_region_means(out_cyc, dims=out_cyc[0].shape[-1], device = 'cpu', dtype=torch.float64)

            fused_op = lambda x:NN.layers[
                current_layer
            ].get_activation_pattern(
                NN.layers[:current_layer].forward(x)
            ).cpu().type(torch.float32)


            q = graph._batched_gpu_op(method=fused_op,
                            data=means,
                            batch_size=fwd_batch_size,
                            out_size=(
                                means.shape[0],torch.prod(NN.layers[current_layer].output_shape)
                            ),
                            dtype= torch.float32,
                            workers=n_workers,
                )


            del means

            Wb =  NN.layers[current_layer].get_weights(dtype=torch.float32).cuda()
            Abw = Abw.type(torch.float32)

            dloader = torch.utils.data.DataLoader(Abw,
                                          pin_memory=True,
                                          batch_size=Abw_batch_size,
                                          num_workers=n_workers,
                                          shuffle=False,
                                          sampler=out_idx,
                                          drop_last=False)

            out_Abw = torch.zeros(len(out_idx),Wb.shape[0],Abw.shape[-1], device='cpu', dtype=torch.float32)

            start = 0
            for in_batch in tqdm.tqdm(dloader,desc='Get Abw',total=len(dloader)):

                end = start+in_batch.shape[0]

                out_batch = utils.get_Abw(
                        q = q[start:end].cuda(),
                        Wb = Wb.to_dense(),
                        incoming_Abw = in_batch.cuda()
                            ) 

                out_Abw[start:end] = out_batch.cpu()
                start = end

            Abw = out_Abw.type(torch.float64)

        del Wb, out_Abw

    elapsed_time = time.time()-start_time
    print(f'Time elapsed {elapsed_time/60:.3f} minutes')
    
    try:
        hyp2input,endpoints = to_next_layer_partition_batched(out_cyc, Abw, -1, NN,
                                        dtype=torch.float64, device='cuda',
                                        batch_size=batch_size,
                                        fwd_batch_size=fwd_batch_size)
    except:
        endpoints = [None]
        
    return out_cyc,endpoints
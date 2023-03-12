import torch
import numpy as np

import networkx as nx
import igraph as ig
import graph_tool as gt
from graph_tool import topology

from splinecam.utils import verify_collinear, get_region_means, get_Abw

import tqdm

@torch.jit.script
def make_line_2D(vert1,vert2):
    '''
    make 2D lines
    vert1: batchsize x 2
    vert2: batchsize x 2
    '''
    x1x2 = vert1[:,0]-vert2[:,0]
    y1y2 = vert1[:,1]-vert2[:,1]
    b = vert1[:,1]*x1x2-vert1[:,0]*y1y2
    return torch.hstack((y1y2[...,None],-x1x2[...,None],b[...,None]))

@torch.jit.script
def find_intersection_2D(line1, line2, eps : float = 1e-7, verify : bool = False):
    '''
    lines1: batch_size x 3
    lines2: batch_size x 3
    '''
    
    Ab = torch.cat(
        (torch.reshape(line1,shape=(line1.shape[0],1,line1.shape[1])),
         torch.reshape(line2,shape=(line2.shape[0],1,line2.shape[1]))),
        dim=1
    )
    
    # Ax + b = 0 -> Ax = -b
    v = torch.linalg.solve(Ab[...,:-1], -Ab[...,-1])
    
    flag = False
    if verify:
        
        flag = torch.allclose(
            torch.bmm(Ab[...,:-1],v[...,None]),
            -Ab[...,-1][...,None],
            atol=eps,
            rtol=0.,
            equal_nan=False
        )
    
    return v,flag

@torch.jit.script
def get_intersection_pattern(poly,hyps):
    pre_act = (hyps[...,:-1] @ poly.T + hyps[...,-1:]).T
    q = (pre_act>0)*1
    return q


@torch.jit.script
def edge_hyp_intersections(qT,poly,hyps):
    '''
    Intersection cases for q \in {1,0}
    1. intersects two edges of polytope: two change of symbols for two different set of edges; two changes in q in a row
    4. intersects one vertex: only one zero and no change of symbol on two sides of zero; two consecutive changes in q
    '''
    
    # find where intersection patters change; 
    ## add vertex intersection check (cases 2-4)
    mask = qT[...,:-1] != qT[...,1:] ## happens outside as well
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    
    # index for hyp and point pairs
    hyp_v1_v2_idx = torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])
    
    return hyp_v1_v2_idx

@torch.jit.script
def vertex_order_along_line_batched(endpoint1,endpoint2,v):
    '''
    Returns the ordered index of the vertex sequence `v` from `endpoint1` to `endpoint2`
    endpoint1,endpoint2: batchsize x 1 x 2
    v:                   batchsize x N x 2
    '''
    v = torch.cat([
        endpoint1,endpoint2,v
    ],dim=1)
    
    dim_to_sort = torch.argmax(v.std(0))
    
    idx = v[:,:,dim_to_sort].argsort()
    
    endpoint_match = idx[:,0] == 0
    
    if torch.all(endpoint_match):
        pass
    else:
        flip_idx = torch.where(
            torch.logical_not(endpoint_match)
        )[0]
        
        idx[flip_idx,:] = torch.flip(idx[flip_idx,:],dims=(1,))
        
    return idx[:,1:-1]-2

@torch.jit.script
def vertex_order_along_line(endpoint1,endpoint2,v):
    '''
    Returns the ordered index of the vertex sequence `v` from `endpoint1` to `endpoint2`
    endpoint1,endpoint2: 1 x 2
    v:                   N x 2
    '''
    v = torch.cat([
        endpoint1,endpoint2,v
    ],dim=0)
    
    dim_to_sort = torch.argmax(v.std(0)) ## to avoid axis aligned lines
    
    idx = v[:,dim_to_sort].argsort()
    
    endpoint_match1 = idx[0] == 0
    endpoint_match2 = idx[-1] == 1
    endpoint_match_rev1 = idx[0] == 1
    endpoint_match_rev2 = idx[-1] == 0
    
    if (endpoint_match1 and endpoint_match2):
        pass
    
    elif endpoint_match_rev1 and endpoint_match_rev2:
        idx = torch.flip(idx,dims=(0,))
        
    else:
        print('sorting dimension ',dim_to_sort)
        print('sorted vertices', v[idx])
        print('sorted idx ',idx)
        raise ValueError('sorting issue')
        
    return idx[1:-1]-2


# @torch.jit.script
def order_vertices_poly(v,hyp_v1_v2_idx,poly,node_names):
    
    hyp_v1_v2_idx = hyp_v1_v2_idx.clone()
    v = v.clone()
    poly = poly.clone()
    node_names = node_names.clone()
    
    v_new = []
    hyp_v1_v2_idx_new = []
    node_names_new = []
    
    for ii in torch.unique(hyp_v1_v2_idx[:,1]):
        
        # vertices with same start node
        mask = hyp_v1_v2_idx[:,1] == ii
        adj = hyp_v1_v2_idx[mask].clone()
        verts = v[mask].clone()
        nodes = node_names[mask].clone()
        
        # order along line, ordered from poly[ii] to poly[ii+1] 
        idx = vertex_order_along_line(poly[ii][None,...],
                                      poly[ii+1][None,...],
                                      verts)
        
        v_new.append(verts[idx])
        hyp_v1_v2_idx_new.append(adj[idx])
        node_names_new.append(nodes[idx])
        
    return v_new,hyp_v1_v2_idx_new,node_names_new


def add_line_to_graph(G,
                      node_names,start,end,
                      v,
                      line_name='',
                      layer_name=-1
                     ):
    '''
    G graph to add new edges to
    names: names for new nodes
    start: start node currently in graph
    end: end node currently in graph
    v: new node vertices
    '''
    
    try:
        node_names = node_names.numpy()
        start = np.int64(start.numpy().squeeze())
        end = np.int64(end.numpy().squeeze())
        line_name = np.int64(line_name.numpy().squeeze())
    except:
        pass
    
    v = v.cpu()
    
    [ G.add_node(o,v=vt) for o,vt in zip(node_names, v) ]
    
    G.add_edge(start,node_names[0],layer=layer_name,
               hyp=line_name
              )
        
    [
        G.add_edge(src,dst,
                   layer=layer_name,hyp=line_name,
                  ) for src,dst in zip(node_names[:-1],node_names[1:])
    ]
        
    G.add_edge(node_names[-1],end,
               layer=layer_name,hyp=line_name)
    
    return

def set_bidirectional(G):
    '''
    Make graph-tool graph bidirectional
    '''
    
    G.set_directed(True)

    for e in G.get_edges():
        G.add_edge(e[1],e[0])
        
def _find_cycles(V,start_edge):
    '''
    Given a bidirectional graph and a starting edge find cycles from that edge
    '''
    
    edge_list_remove = [[v for v in start_edge]]
    
    ## if edge is a boundary edge
    if not V.ep['layer'][start_edge] == -1:
        raise ValueError('start_edge must be a boundary edge')
    
    V.remove_edge(start_edge)
    
    out_cycles = []  
    
    for each_edge in edge_list_remove:
        
        remove_q = []
        
        vertices = []
        vertex_id = []
        for v in each_edge:
            vertices.append(v)
            vertex_id.append(V.vertex_index[v])
        
        ## if no way in and no way out
        
        if not (V.get_in_degrees(
            [vertex_id[1]]
        )[0]>1) and (V.get_out_degrees(
            [vertex_id[0]]
        )[0]>1):
            
            continue
                                                  
        if V.edge(*vertices) is None and V.edge(vertices[1],vertices[0]) is None:
            continue
      
        remove_q.append(V.edge(vertices[1],vertices[0])) # remove opposite path as well
        vs,es = topology.shortest_path(V,
                 source=vertices[0],
                 target=vertices[1],
#                  weights=gT.ep['len'] ## for dijkstra; bfs faster
                )
                            
        out_cycles.append([V.vertex_index[each] for each in vs])
   
        for e in es:
            v = [v for v in e]
            if V.ep['layer'][e] == -1:
                remove_q.append(e)
                remove_q.append(V.edge(v[1],v[0]))
            else:
                remove_q.append(e)
                edge_list_remove.append(v)
        
        for each in remove_q:
            try:
                V.remove_edge(each)
            except:
                pass
                
#         [V.remove_edge(each) for each in remove_q] #only remove new edges
        
    return out_cycles
        
        
def find_cycles_in_graph(G,return_coordinates=False):
    '''
    Given a graph-tool graph, find the cycles present in it
    '''
    
    # deep copy
    V = gt.Graph(G)

    set_bidirectional(V)
    
    # tradeoff time complexity for space complexity
    V.set_fast_edge_removal()
    
    ## find boundary edge to start from
    for e in V.edges():
        if V.ep['layer'][e] == -1:
            break
    
    cycles = _find_cycles(V,e)
    
    cycles = [each for each in cycles if len(each)>1]
    
    if return_coordinates:
        cycles = cycle_nodes2vertices(V,cycles)
    
    return cycles

def cycle_nodes2vertices(V,cycles,dcast=np.asarray):
    '''
    Get vertices for each cycles
    '''
    
    cycles = [[dcast(
            V.vp['v'][V.vertex(v)]
        )  for v in each_cycle] for each_cycle in cycles]
    
    return cycles

def create_poly_hyp_graph(poly, hyps, q=None, hyp_endpoints=None, dtype=torch.float64, verify=True):

    G = nx.Graph()
    
#     hyps = layer.get_weights().type(dtype)
#     poly = poly.type(dtype)
    
    redundant_vert_id = len(poly)-1

    poly_node_idx = np.asarray(list(range(len(poly)-1))+[0])
    # poly_node_idx = torch.from_numpy(poly_node_idx).type(torch.int)
    
    poly_hyp_idx = np.asarray(range(len(poly)-1))
    # poly_hyp_idx = torch.from_numpy(poly_hyp_idx).type(torch.int)

    # add nodes as vertices to graph
    [G.add_node(o,v=v.cpu()) for o,v in zip(poly_node_idx[:-1], poly[:-1])]

#     V = G.copy()
#     # add edges (ONLY add edges that dont intersect)
#     [V.add_edge(src,dst,layer=-1,hyp=hyp) for src,dst,hyp in zip(
#         poly_node_idx[:-1],poly_node_idx[1:],poly_hyp_idx
#     )]

    # pos = dict(zip(poly_node_idx,[V.nodes[each]['v'] for each in poly_node_idx]))
    # nx.draw(V,pos=pos)


    new_node_start = poly_node_idx[-2]+1
    node_counter = 0

    ### find hyp and edge intersections, add to graph
    
    # create lines and check intersections
    
    if q is None: 
        q = get_intersection_pattern(poly,hyps)

    no_inter_idx = torch.where(torch.prod(q[:-1] == q[1:],axis=1))[0].cpu()
    
    ## if multiple edges are not intersected
    if len(no_inter_idx)>1:
    
        [
            G.add_edge(src,dst,layer=-1,hyp=hyp) for src,dst,hyp in zip(
            poly_node_idx[no_inter_idx],poly_node_idx[no_inter_idx+1],poly_hyp_idx[no_inter_idx])
        ]
    
    ## if one edge is not intersected
    elif len(no_inter_idx)==1:
        
        G.add_edge(poly_node_idx[no_inter_idx],
                   poly_node_idx[no_inter_idx+1],
                   layer=-1,
                   hyp=poly_hyp_idx[no_inter_idx])
    
    ## if all edges are intersected
    else:
        pass
    
    # get intersecting hypidx and associated vertex idx
    hyp_v1_v2_idx = edge_hyp_intersections(q.T,poly,hyps)
    
    # make polytope lines
    poly_lines = make_line_2D(poly[hyp_v1_v2_idx[:,1]],poly[hyp_v1_v2_idx[:,2]])

    # find intersections
#     if hyp_endpoints is None:
        
    poly_int_hyps = hyps[hyp_v1_v2_idx[:,0]]
    v,flag = find_intersection_2D(poly_lines,
                                  poly_int_hyps,
                                  verify=verify)

    v = v.type(poly_lines.type())
    
    if verify:
        
        assert flag
    
        flag = verify_collinear(v,
                                poly[hyp_v1_v2_idx[:,1]],
                                poly[hyp_v1_v2_idx[:,2]]
                                )

        assert flag

    hyp_endpoints = v.reshape(-1,2,v.shape[-1]) ## 

#     else:
        
#         v = hyp_endpoints.reshape(-1,hyp_endpoints.shape[-1])
#         if v.shape[0] != hyp_v1_v2_idx.shape[0]:
#             print(v)
#             print(hyp_v1_v2_idx)

    ### add to graph

    # indices for new nodes
    new_node_idx = torch.from_numpy(
        np.asarray(range(new_node_start,new_node_start+v.shape[0]))
    ).type(torch.int)

    # create list of ordered vertices
    v_collect, hyp_v1_v2_idx_collect, node_names_collect = order_vertices_poly(v,hyp_v1_v2_idx,poly,new_node_idx)
    
    for v_set,hyp_v1_v2_idx_set,node_names in zip(v_collect,hyp_v1_v2_idx_collect,node_names_collect):    

        add_line_to_graph(
            G=G,
            node_names = node_names,
            start = poly_node_idx[hyp_v1_v2_idx_set[0,1]],
            end = poly_node_idx[hyp_v1_v2_idx_set[0,2]],
            v = v_set,
            line_name = poly_hyp_idx[hyp_v1_v2_idx_set[0,1]],
            layer_name = -1 ## still previous layer
        )
    
    
#     pos = dict([(each,G.nodes[each]['v'].numpy()) for each in G.nodes])
#     nx.draw(G,pos=pos,node_size=50)
    
    uniq_hyp_idx = hyp_v1_v2_idx[::2,0] ## all hyps that intersect
    
#     hyp_endpoints = v.reshape(-1,2,v.shape[-1]) ## 
    hyp_endpoint_nodes = new_node_idx.reshape(-1,2)

    
    # if combination idx empty, just connect the endpoints
    if uniq_hyp_idx.shape[0] <= 1: # < because of no intersection case ##TODO: check why no intersection here for deeper layers
        
        [
            G.add_edge(
                nodes[0],nodes[1],layer=0,hyp=name
            ) for nodes,name in zip(hyp_endpoint_nodes.numpy(),uniq_hyp_idx)
        ]
        
        return G
        
    
    # get combination idx of unique hyperplanes that intersect
    comb_idx,no_inter_idx = create_hyp_combinations(hyps=hyps,
                                       hyp_idx=uniq_hyp_idx,
                                       endpoints=hyp_endpoints)
    
    
    ## if combination idx empty, just connect the endpoints
        
    if no_inter_idx.shape[0] != 0:
        
        if no_inter_idx.shape[0] == 1:
        
            G.add_edge(
                hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()][0],
                hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()][1],
                layer=0,
                hyp=uniq_hyp_idx[no_inter_idx.cpu()]
            )
            
        
        else:
            
            [
                G.add_edge(
                    nodes[0],nodes[1],layer=0,hyp=name
                ) for nodes,name in zip(hyp_endpoint_nodes.numpy()[no_inter_idx.cpu()],
                                        uniq_hyp_idx[no_inter_idx.cpu()])
            ]
    

    if comb_idx.shape[0] == 0:
        return G
        
    v,flag = find_intersection_2D(hyps[comb_idx[:,0]],
                                hyps[comb_idx[:,1]],
                                verify=verify)
    if verify:
        assert flag

    new_node_start = new_node_idx[-1]+1
    new_node_idx = torch.arange(new_node_start,new_node_start+v.shape[0]).type(torch.int)

    for ii,each_hyp in tqdm.tqdm(enumerate(uniq_hyp_idx), desc='iterating hyps', total=len(uniq_hyp_idx)):


        mask = torch.logical_or(comb_idx[:,0] == each_hyp, comb_idx[:,1] == each_hyp)

        if not(torch.sum(mask)): #hyp intersects at vertex, hence
            #came up as uniq hyp but didnt come up as combination
            continue

        verts = v[mask].clone()
        hyp_adj = comb_idx[mask].clone()
        nodes = new_node_idx[mask].clone()

        idx = vertex_order_along_line(
                endpoint1=hyp_endpoints[ii,0][None,...],
                endpoint2=hyp_endpoints[ii,1][None,...],
                v=verts
            )

        verts = verts[idx]  
        hyp_adj = hyp_adj[idx]
        nodes = nodes[idx]

        add_line_to_graph(
            G=G,
            node_names = nodes,
            start = hyp_endpoint_nodes[ii,0],
            end = hyp_endpoint_nodes[ii,1],
            v = verts,
            line_name = each_hyp,
            layer_name = 0 ## coming layer
        )
        
    return G

@torch.jit.script
def hyp2input(hyps,Abw):
    
    hyps = hyps[...,None,:]
    
    hyps_inp = torch.bmm(
        hyps[...,:-1],Abw[...,:-1])
    
    bias_inp = torch.bmm(
        hyps[...,:-1],Abw[...,-1:]) +  hyps[...,-1:]
    
    return torch.cat([hyps_inp,bias_inp],dim=-1)

# @torch.jit.script TODO: Make jittable
def cycles_list2vec(regions, repeat_first : bool = True):
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


@torch.jit.script
def get_edge_hyp_intersections(vec_cyc,hyp_v1_v2_idx,hyps_input):
    
    vec_cyc = vec_cyc.type(torch.float64)
    
    
    poly_lines = make_line_2D(
    vec_cyc[hyp_v1_v2_idx[:,1]],
    vec_cyc[hyp_v1_v2_idx[:,2]]
    )
    
    v,flag = find_intersection_2D(poly_lines,
                              hyps_input,
                              verify=False)
    
#     if not flag:
#         print('intersection flag false')
    
#     flag = verify_collinear(v,
#                         vec_cyc[hyp_v1_v2_idx[:,1]],
#                         vec_cyc[hyp_v1_v2_idx[:,2]]
#                         )
    
#     if not flag:
#         print('collinear flag false')
    
    return v


@torch.jit.script
def create_hyp_combinations(hyps,hyp_idx,endpoints):
    '''
    find which hyps 
    '''
    ## make sure the number of hyps and endpoints are the same
    assert len(hyp_idx) == endpoints.shape[0]
    
    ## check intersection for endpoints and hyps
    q = get_intersection_pattern(endpoints.reshape(-1,endpoints.shape[-1]),hyps[hyp_idx])
    q = q.reshape(-1,2,hyp_idx.shape[0])
    
    ## only consider endpoints which change pattern 
#     q = np.logical_xor.reduce(q,axis=1)
    q = torch.logical_xor(
                            q[:,0,:].reshape(-1),
                            q[:,1,:].reshape(-1)
                        ).view(q.shape[0],q.shape[-1])
    
    # remove upper triangular and diagonal
    mask = torch.tril(torch.ones_like(q)) 
    q *= torch.logical_not(
        torch.eye(q.shape[0]).to(q.device)
    ) 
    no_inter_idx = torch.where(q.sum(1) == 0)[0]
    q *= mask
    
    # get combination
    loc = torch.where(q)
    comb_idx = torch.stack([
        hyp_idx[loc[0]],hyp_idx[loc[1]]
    ]).T
    
    return comb_idx,no_inter_idx


@torch.no_grad()
def to_next_layer_partition(cycles, Abw, current_layer, NN, dtype=torch.float64, device='cuda'):
    
    vec_cyc,cyc_idx,ends = cycles_list2vec(cycles)
    cycles_next = NN.layers[:current_layer].forward(vec_cyc.to(device))
    q = NN.layers[current_layer].get_intersection_pattern(cycles_next)
    
    ## edge intersections. remove between cycles
    mask = q.T[...,:-1] != q.T[...,1:]
    mask = mask.cpu()
    mask[:,(ends-1)[:-1]] = False
    
    if mask.sum() == 0:
        return cycles, torch.arange(len(cycles))
    
    
    ## get indices for hyps-vertex-cycle triads
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    hyp_vert_cyc_idx = torch.hstack([hyp_vert_idx,cyc_idx[hyp_vert_idx[:,1:]]])
    
    ## assert all cycles occur twice in order
    assert torch.all(hyp_vert_cyc_idx[::2,2] == hyp_vert_cyc_idx[1::2,2])
    
    ## query hyps, only get rows which intersect, create idx map
    inter_hyps_idx = torch.unique(hyp_vert_cyc_idx[:,0])
    hyps = NN.layers[current_layer].get_weights(row_idx=inter_hyps_idx)
    hyp_idx_map = torch.ones(q.shape[1],dtype=torch.int64)*(hyps.shape[0]+100) ## initialize with idx out of range
    hyp_idx_map[inter_hyps_idx] = torch.arange(hyps.shape[0], dtype=torch.int64)
    
    ## bring hyps to corresponding cycle inputs
    hyps_input = hyp2input(
        hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]].to(device), ## hyps that intersect
        Abw[hyp_vert_cyc_idx[::2,2]].to(device) ## corresponding region Abw
    )[:,0,:]
    
    
    ## get intersection with all cycle edges
    hyp_v1_v2_idx= torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])
    v = get_edge_hyp_intersections(
        vec_cyc = vec_cyc.to(device),
        hyps_input = torch.repeat_interleave(hyps_input,2,dim=0).to(device),
        hyp_v1_v2_idx = hyp_v1_v2_idx
    )
    
    hyp_endpoints = v.reshape(-1,2,v.shape[-1])
    
    ## iterate over each region and obtain new regions
    uniq_cycle_idx = torch.unique(hyp_vert_cyc_idx[:,-1])
    
    res_regions = []
    new_cyc_idx = []
    
    ## for each intersected cycle, find new regions
    for target_cycle_idx in tqdm.tqdm(uniq_cycle_idx):
        
        vert_mask = cyc_idx==target_cycle_idx
        hyp_mask = hyp_vert_cyc_idx[::2,-1] == target_cycle_idx

        
        G = create_poly_hyp_graph(
            poly = vec_cyc[vert_mask].to(device),
            hyps = hyps_input[hyp_mask].to(device),
            hyp_endpoints = hyp_endpoints[hyp_mask].to(device),
            dtype = dtype
        )
        
        G = ig.Graph.from_networkx(G)

        G = G.to_graph_tool(
            vertex_attributes={'v':'vector<float>'},
            edge_attributes={'layer':'int','hyp':'int'}
        )
        
        if current_layer == 1:
            print('Finding layer 1 regions')
        
        cycles_new = find_cycles_in_graph(G,return_coordinates=False)

        cycles_new = cycle_nodes2vertices(
            G,
            cycles_new,
            dcast=lambda x: torch.from_numpy(
                np.asarray(x),
            ).type(dtype),
        )
        cycles_new = [torch.vstack(each) for each in cycles_new]
        new_cyc_idx += [target_cycle_idx for i in range(len(cycles_new))]
        
        res_regions += cycles_new
    
    
    ## add cycles that were not intersected
    non_int_cyc_idx = [each_idx for each_idx in torch.arange(len(cycles)) if each_idx not in uniq_cycle_idx]
    
    res_regions += [cycles[each_idx] for each_idx in non_int_cyc_idx]
    new_cyc_idx += non_int_cyc_idx
    
    return res_regions, new_cyc_idx

def _batched_gpu_op(method, data, batch_size, out_size, dtype=torch.float32, workers=2, out_device='cpu'):
    
    dataloadr = torch.utils.data.DataLoader(data,
                                      pin_memory=False,
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      shuffle=False,
                                      drop_last=False)
    
    ##malloc
    out = torch.zeros(out_size, device=out_device, dtype=dtype)
    
    start = 0
    for in_batch in dataloadr:
        
        end  = start+in_batch.shape[0]
        out_batch = method(in_batch.cuda())
        out[start:end] = out_batch.to(out_device)
        start = end

    return out

class util_dataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
        self._len = self.data1.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


def _batched_gpu_op_2(method, data1, data2, batch_size, out_size, dtype=torch.float32, workers=2):
    
    assert data1.shape[0] == data2.shape[0]
    
    dataloadr = torch.utils.data.DataLoader(util_dataset(data1,data2),
                                      pin_memory=True,
                                      batch_size=batch_size,
                                      num_workers=workers,
                                      shuffle=False,
                                      drop_last=False)
    
    ##malloc
    out = torch.zeros(out_size, device='cpu', dtype=dtype)
    
    start = 0
    for in_batch1,in_batch2 in dataloadr:
        
        end  = start+in_batch1.shape[0]
        out_batch = method(in_batch1.cuda(),in_batch2.cuda())
        out[start:end] = out_batch.cpu()
        start = end

    return out


@torch.no_grad()
def to_next_layer_partition_batched(cycles, Abw, current_layer, NN,
                                    dtype=torch.float64, device='cuda',
                                    batch_size=-1, fwd_batch_size=-1, workers=2):
    
    if batch_size == -1: ## revert to non-batched
        res_regions, new_cyc_idx = to_next_layer_partition(
            cycles, Abw, current_layer, NN, dtype, device
        )
        return res_regions, new_cyc_idx
    
    vec_cyc,cyc_idx,ends = cycles_list2vec(cycles)
    
#     cycles_next = NN.layers[:current_layer].forward(vec_cyc.to(device))
#     q = NN.layers[current_layer].get_intersection_pattern(cycles_next)
    
    fused_op = lambda x: NN.layers[current_layer].get_intersection_pattern(
        NN.layers[:current_layer].forward(x))
    
        
    q = _batched_gpu_op(fused_op, 
                        vec_cyc,
                        workers = workers,
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
        return cycles, torch.arange(len(cycles))
    
#     del  q
#     del cycles_next
    
    ## get indices for hyps-vertex-cycle triads
    hyp_vert_idx = torch.vstack(torch.where(mask)).T
    hyp_vert_cyc_idx = torch.hstack([hyp_vert_idx,cyc_idx[hyp_vert_idx[:,1:]]])
    
    ## assert all cycles occur twice in order
    assert torch.all(hyp_vert_cyc_idx[::2,2] == hyp_vert_cyc_idx[1::2,2])
    
    ## query hyps, only get rows which intersect, create idx map
    inter_hyps_idx = torch.unique(hyp_vert_cyc_idx[:,0])
    hyps = NN.layers[current_layer].get_weights(row_idx=inter_hyps_idx).cpu()
    hyp_idx_map = torch.ones(n_hyps,dtype=torch.int64)*(hyps.shape[0]+100) ## initialize with idx out of range
    hyp_idx_map[inter_hyps_idx] = torch.arange(hyps.shape[0], dtype=torch.int64)
    
    ## bring hyps to corresponding cycle inputs
    
    hyps_input = _batched_gpu_op_2(
        method = hyp2input,
        data1 = hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]],
        data2 = Abw[hyp_vert_cyc_idx[::2,2]],
        batch_size = batch_size,
        out_size = (hyp_vert_cyc_idx[::2,0].shape[0],1,3),
        dtype = dtype,
        workers = workers
    )[:,0,:]
        
    
#     hyps_input = hyp2input(
#         hyps[hyp_idx_map[hyp_vert_cyc_idx[::2,0]]].to(device), ## hyps that intersect
#         Abw[hyp_vert_cyc_idx[::2,2]].to(device) ## corresponding region Abw
#     )[:,0,:]
    
    
    ## get intersection with all cycle edges
#     hyp_v1_v2_idx= torch.hstack([hyp_vert_idx,hyp_vert_idx[:,-1:]+1])
#     v = get_edge_hyp_intersections(
#         vec_cyc = vec_cyc.to(device),
#         hyps_input = torch.repeat_interleave(hyps_input,2,dim=0).to(device),
#         hyp_v1_v2_idx = hyp_v1_v2_idx
#     )
    
#     hyp_endpoints = v.reshape(-1,2,v.shape[-1])
    
    ## iterate over each region and obtain new regions
    uniq_cycle_idx = torch.unique(hyp_vert_cyc_idx[:,-1])
    
    res_regions = []
    new_cyc_idx = []
    
    ## for each intersected cycle, find new regions
    for target_cycle_idx in tqdm.tqdm(uniq_cycle_idx, desc='Iterating regions'):
        
        vert_mask = cyc_idx==target_cycle_idx
        hyp_mask = hyp_vert_cyc_idx[::2,-1] == target_cycle_idx
        
        G = create_poly_hyp_graph(
            poly = vec_cyc[vert_mask].to(device),
            hyps = hyps_input[hyp_mask].to(device),
#             hyp_endpoints = hyp_endpoints[hyp_mask].to(device),
            dtype = dtype
        )
        
#         import networkx as nx
#         pos = dict([(each,G.nodes[each]['v']) for each in G.nodes])
#         nx.draw(G,pos=pos)
        
        G = ig.Graph.from_networkx(G)

        G = G.to_graph_tool(
            vertex_attributes={'v':'vector<float>'},
            edge_attributes={'layer':'int','hyp':'int'}
        )
        
        if current_layer == 1:
            print('Finding regions from first layer graph')
        
        cycles_new = find_cycles_in_graph(G,return_coordinates=False)

        cycles_new = cycle_nodes2vertices(
            G,
            cycles_new,
            dcast=lambda x: torch.from_numpy(
                np.asarray(x),
            ).type(dtype),
        )
        cycles_new = [torch.vstack(each) for each in cycles_new]
        
        new_cyc_idx += [target_cycle_idx for i in range(len(cycles_new))]
        
        res_regions += cycles_new
    
    
    ## add cycles that were not intersected
    non_int_cyc_idx = [each_idx for each_idx in torch.arange(len(cycles)) if each_idx not in uniq_cycle_idx]
    
    res_regions += [cycles[each_idx] for each_idx in non_int_cyc_idx]
    new_cyc_idx += non_int_cyc_idx
    
    return res_regions, new_cyc_idx

def networkx2graphtool(G):
    
    G = ig.Graph.from_networkx(G)

    G = G.to_graph_tool(
        vertex_attributes={'v':'vector<float>'},
        edge_attributes={'layer':'int','hyp':'int'}
    )
    
    return G


def compute_partitions_with_db(
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

        out_cyc,out_idx = to_next_layer_partition_batched(
            cycles = out_cyc,
            Abw = Abw,
            NN = NN,
            current_layer = current_layer,
            dtype = torch.float64,
            batch_size=batch_size,
            fwd_batch_size=fwd_batch_size,
        )

        with torch.no_grad():

            means = get_region_means(out_cyc, dims=out_cyc[0].shape[-1], device = 'cpu', dtype=torch.float64)

            fused_op = lambda x:NN.layers[
                current_layer
            ].get_activation_pattern(
                NN.layers[:current_layer].forward(x)
            ).cpu().type(torch.float32)


            q = _batched_gpu_op(method=fused_op,
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

                out_batch = get_Abw(
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


def compute_partitions(
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

        out_cyc,out_idx = to_next_layer_partition_batched(
            cycles = out_cyc,
            Abw = Abw,
            NN = NN,
            current_layer = current_layer,
            dtype = torch.float64,
            batch_size=batch_size,
            fwd_batch_size=fwd_batch_size,
        )

        with torch.no_grad():

            means = get_region_means(out_cyc, dims=out_cyc[0].shape[-1], device = 'cpu', dtype=torch.float64)

            fused_op = lambda x:NN.layers[
                current_layer
            ].get_activation_pattern(
                NN.layers[:current_layer].forward(x)
            ).cpu().type(torch.float32)


            q = _batched_gpu_op(method=fused_op,
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

                out_batch = get_Abw(
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

    return out_cyc,endpoints
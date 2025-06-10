import dgl
import torch
import random
import textwrap
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dgl.subgraph import khop_in_subgraph
from itertools import count
from heapq import heappop, heappush
from sklearn.metrics import roc_auc_score
import community
from cdlib import algorithms, evaluation
import igraph as ig
from collections import Counter


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_args(args):
    for k, v in vars(args).items():
        print(f'{k:25} {v}')
        
def set_config_args(args, config_path, dataset_name, model_name=''):
    with open(config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)[dataset_name]
        if model_name:
            config = config[model_name]

    for key, value in config.items():
        setattr(args, key, value)
    return args
def shifted_sigmoid(x, shift):
    return 1 / (1 + torch.exp(-(x + shift)))    
    
    
    
    
'''
Model training utils
'''
def idx_split(idx, ratio, seed=0):
    """
    Randomly split `idx` into idx1 and idx2, where idx1 : idx2 = `ratio` : 1 - `ratio`
    
    Parameters
    ----------
    idx : tensor
        
    ratio: float
 
    Returns
    ----------
        Two index (tensor) after split
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def eids_split(eids, val_ratio, test_ratio, seed=0):
    """
    Split `eids` into three parts: train, valid, and test,
    where train : valid : test = (1 - `val_ratio` - `test_ratio`) : `val_ratio` : `test_ratio`
    
    Parameters
    ----------
    eid : tensor
        edge id
        
    val_ratio : float
    
    test_ratio : float

    seed : int

    Returns
    ----------
        Three edge ids (tensor) after split
    """
    train_ratio = (1 - val_ratio - test_ratio)
    train_eids, pred_eids = idx_split(eids, train_ratio, seed)
    val_eids, test_eids = idx_split(pred_eids, val_ratio / (1 - train_ratio), seed)
    return train_eids, val_eids, test_eids

def negative_sampling(graph, pred_etype=None, num_neg_samples=None):
    '''
    Adapted from PyG negative_sampling function
    https://pytorch-geometric.readthedocs.io/en/1.7.2/_modules/torch_geometric/utils/
    negative_sampling.html#negative_sampling

    Parameters
    ----------
    graph : dgl graph
    
    pred_etype : string
        The edge type for prediction

    num_neg_samples : int
    
    Returns
    ----------
        Two negative nids. Nids for src and tgt nodes of the `pred_etype`
    '''
    # src_N: total number of src nodes
    # N (tgt_N): total number of tgt nodes
    # M: total number of possible edges, square of src_N * tgt_N
    # pos_M: number of positive samples (observed edges)
    # neg_M: number of negative samples
    pos_src_nids, pos_tgt_nids = graph.edges(etype=pred_etype)
    if pred_etype is None:
        N = graph.num_nodes()
        M = N * N
    else:
        src_ntype, _, tgt_ntype = graph.to_canonical_etype(pred_etype) 
        src_N, N = graph.num_nodes(src_ntype), graph.num_nodes(tgt_ntype)
        M = src_N * N

    pos_M = pos_src_nids.shape[0]
    neg_M = num_neg_samples or pos_M
    neg_M = min(neg_M, M - pos_M) # incase M - pos_M < neg_M

    # Percentage of edges to opos_tgt_nidsersample, so only need to sample once in most cases
    alpha = abs(1 / (1 - 1.1 * (pos_M / M)))
    size = min(M, int(alpha * neg_M))
    perm = torch.tensor(random.sample(range(M), size))
    
    idx = pos_src_nids * N + pos_tgt_nids
    # mask = torch.from_npos_src_nidsmpy(np.isin(perm, idx.to('cppos_src_nids'))).to(torch.bool)
    mask = torch.isin(perm, idx.to('cpu')).to(torch.bool)
    perm = perm[~mask][:neg_M].to(pos_src_nids.device)

    neg_src_nids = torch.div(perm, N, rounding_mode='floor')
    neg_tgt_nids = perm % N

    return neg_src_nids, neg_tgt_nids


def get_community(g):     
    g_nx = dgl.to_networkx(g.cpu(), node_attrs=[dgl.NTYPE]).to_undirected()
    partition = community.best_partition(g_nx)
    communities = {n: cid for n, cid in partition.items()}
    return communities      

def get_top_communities(community_strengths, community_nodes, N):
    
    total_nodes = 0
    selected_communities = []
    
    
    for community, strength in community_strengths.items():
        if community in community_nodes:
            total_nodes += community_nodes[community]
            selected_communities.append(community)
            
            
            if total_nodes >= N / 2:
                return selected_communities, len(selected_communities)
    
    
    return selected_communities, len(selected_communities)
    


def split_list_into_two_parts(lst):
    total_sum = sum(lst)
    target_sum = total_sum / 2  # 
    current_sum = 0.0
    split_index = 0

    for i, num in enumerate(lst):
        current_sum += num
        if current_sum >= target_sum:
            split_index = i + 1
            break

    part1 = lst[:split_index]
    part2 = lst[split_index:]

    # 
    if part1 and part2:  # 
        threshold = (part1[-1] + part2[0]) / 2
    else:
        threshold = None  

    return part1, part2, threshold
    
def node_strength(G,community_dict):
    
    pagerank = nx.pagerank(G)
    
   
    community_pagerank_sum = {}
    community_node_count = {}
    
    for node, pr in pagerank.items():
        comm = community_dict[node]
        if comm in community_pagerank_sum:
            community_pagerank_sum[comm] += pr
            community_node_count[comm] += 1
        else:
            community_pagerank_sum[comm] = pr
            community_node_count[comm] = 1
    
    # 
    community_pagerank_avg = {}
    for comm in community_pagerank_sum:
        avg = community_pagerank_sum[comm] / community_node_count[comm]
        community_pagerank_avg[comm] = avg
      
    return community_pagerank_avg
    
def all_community_strength(community_strength,community_pagerank_avg):
    
    for comm, avg_pr in community_pagerank_avg.items():
        if comm in community_strength:
            community_strength[comm] += avg_pr
        else:
            
            community_strength[comm] = avg_pr    
    return community_strength
    
              
def louvain_partition(graph):

    q = 0
    ghomo = dgl.to_homogeneous(graph)
    G = dgl.to_networkx(ghomo.cpu(), node_attrs=[dgl.NTYPE]).to_undirected()
    
    total_edges = G.size()
    degrees = dict(G.degree())
    
    #print(110,is_weighted(G))

    print(">> Louvain Clustering...")
    partition = community.best_partition(G)
    
    
    N=len(G.nodes())


    

    community_edges = {}
    for node, com in partition.items():
        if com not in community_edges:
            community_edges[com] = 0
        community_edges[com] += len([n for n in G.neighbors(node) if partition[n] == com])     
    
    community_edges=dict(sorted(community_edges.items()))
    

    
  
    
    
    
    community_degree_sums = {}
    for node, com in partition.items():
        if com not in community_degree_sums:
            community_degree_sums[com] = 0
        community_degree_sums[com] += degrees[node]
    community_degree_sums=dict(sorted(community_degree_sums.items()))
    

    community_strengths = {}
    
    node_community_strengths = {}
    for node, com in partition.items():
        if community_edges.get(com, 0) == 0:
            community_strengths[com] = q
            node_community_strengths[node] = q
        else:
            community_strengths[com] = community_edges[com]/ total_edges -  (community_degree_sums[com]/(2 *total_edges)) ** 2
            node_community_strengths[node] = community_strengths[com]




    #print(111,community_strengths)
    community_pagerank_avg=node_strength(G,partition)
    #print(222,community_pagerank_avg)
    a_community_strengths = all_community_strength(community_strengths,community_pagerank_avg)
    #print(333,a_community_strengths)

    a_community_strengths=dict(sorted(a_community_strengths.items(), key=lambda x: x[1], reverse=True))
    strengths_values=list(a_community_strengths.values())
    #print(444,strengths_values)
    strenthpart,weakpart,threshold=split_list_into_two_parts(strengths_values)
    #print(555,strenthpart,weakpart,threshold)

    
    
    membership = []
    for node in partition.keys():
        membership.append(partition[node])

    membership = np.array(membership)
    num_cluster = np.max(membership) + 1
    print(">> Louvain Clustering Finished, {:d} communities detected.".format(num_cluster))
    #print(membership)
    #print(l)
    
    
    
   
    
    
    

    return membership,community_strengths, node_community_strengths,threshold
    
      
    
def get_hetero_node_community_info(ghetero):
    homo_nids_to_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)

    homo_membership,community_strengths, node_community_strengths,threshold = louvain_partition(ghetero)
    
    
    
      
    hetero_membership = {}
    for homo_nid, community_id in enumerate(homo_membership):
        hetero_nid = homo_nids_to_hetero_nids[homo_nid]
        hetero_membership[hetero_nid] = community_id
    #print(222,hetero_membership)
    
    
    
    
    
    hetero_node_community_strengths = {}
    for homo_nid, node_community_strength in enumerate(node_community_strengths.values()):
        hetero_nid = homo_nids_to_hetero_nids[homo_nid]
        hetero_node_community_strengths[hetero_nid] = node_community_strength
    #print(333,hetero_node_community_strengths) 
    
    

    
    edges_intro_id=[]
    edges_inter_id=[]   
    
    for start_ntype, e_etype, end_ntype in ghetero.canonical_etypes:
        u, v = ghetero.edges(etype=e_etype)
        M = u.shape[0] # number of directed edges
        eids = torch.arange(M)
        u=u.tolist()
        v=v.tolist()
        edges=[(x, y) for x, y in zip(u, v)]           
        for index, (i,j) in enumerate(edges):
            if  hetero_node_community_strengths[(start_ntype,i)]==  hetero_node_community_strengths[(end_ntype,j)]:
              edges_intro_id.append(index)
            else:
              edges_inter_id.append(index)
    internal_pro =len(edges_intro_id)
    cross_pro=len(edges_inter_id)
    print(1111,internal_pro,cross_pro)
    
    
    return hetero_membership,community_strengths, hetero_node_community_strengths,internal_pro ,cross_pro 
    
    

        
        

'''
DGL graph manipulation utils
'''
def get_homo_nids_to_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to the node ids of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_hetero_nids : dict
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    homo_nids_to_hetero_nids = dict(zip(homo_nids, hetero_nids))
    return homo_nids_to_hetero_nids

def get_homo_nids_to_ntype_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to tuples as (node type, node id) of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_ntype_hetero_nids : dict
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    ntypes = ghetero.ntypes
    # This line relies on the default order of ntype_ids is the order in ghetero.ntypes
    ntypes = [ntypes[i] for i in ghomo.ndata[dgl.NTYPE]] 
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    ntypes_hetero_nids = list(zip(ntypes, hetero_nids))
    homo_nids_to_ntype_hetero_nids = dict(zip(homo_nids, ntypes_hetero_nids))
    return homo_nids_to_ntype_hetero_nids

def get_ntype_hetero_nids_to_homo_nids(ghetero):
    '''
    Create a dictionary mapping tuples as (node type, node id) of the input heterogeneous graph
    to the node ids of the homogeneous version of the input graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    ntype_hetero_nids_to_homo_nids : dict
    '''
    tmp = get_homo_nids_to_ntype_hetero_nids(ghetero)
    ntype_hetero_nids_to_homo_nids = {v: k for k, v in tmp.items()}
    return ntype_hetero_nids_to_homo_nids

def get_ntype_pairs_to_cannonical_etypes(ghetero, pred_etype='likes'):
    '''
    Create a dictionary mapping tuples as (source node type, target node type) to 
    cannonical edge types. Edges wity type `pred_etype` will be excluded.
    A helper function for path finding.
    Only works if there is only one edge type between any pair of node types.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
      
    pred_etype : string
        The edge type for prediction

    Returns
    ----------
    ntype_pairs_to_cannonical_etypes : dict
    '''
    ntype_pairs_to_cannonical_etypes = {}
    for src_ntype, etype, tgt_ntype in ghetero.canonical_etypes:
        if etype != pred_etype:
            ntype_pairs_to_cannonical_etypes[(src_ntype, tgt_ntype)] = (src_ntype, etype, tgt_ntype)
    return ntype_pairs_to_cannonical_etypes

def get_num_nodes_dict(ghetero):
    '''
    Create a dictionary containing number of nodes of all ntypes in a heterogeneous graph
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    Returns 
    ----------
    num_nodes_dict : dict
        key=node type, value=number of nodes
    '''
    num_nodes_dict = {}
    for ntype in ghetero.ntypes:
        num_nodes_dict[ntype] = ghetero.num_nodes(ntype)    
    return num_nodes_dict

def remove_all_edges_of_etype(ghetero, etype):
    '''
    Remove all edges with type `etype` from `ghetero`. If `etype` is not in `ghetero`, do nothing.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    etype : string or triple of strings
        Edge type in simple form (string) or cannonical form (triple of strings)
    
    Returns 
    ----------
    removed_ghetero : heterogeneous dgl graph
        
    '''
    etype = ghetero.to_canonical_etype(etype)
    if etype in ghetero.canonical_etypes:
        eids = ghetero.edges('eid', etype=etype)
        removed_ghetero = dgl.remove_edges(ghetero, eids, etype=etype)
    else:
        removed_ghetero = ghetero
    return removed_ghetero

def hetero_src_tgt_khop_in_subgraph(src_ntype, src_nid, tgt_ntype, tgt_nid, ghetero, k):
    '''
    Find the `k`-hop subgraph around the src node and tgt node in `ghetero`
    The output will be the union of two subgraphs.
    See the dgl `khop_in_subgraph` function as a referrence
    https://docs.dgl.ai/en/0.9.x/generated/dgl.khop_in_subgraph.html
    
    Parameters
    ----------
    src_ntype: string
        source node type
    
    src_nid : int
        source node id

    tgt_ntype: string
        target node type

    tgt_nid : int
        target node id

    ghetero : heterogeneous dgl graph

    k: int
        Number of hops

    Return
    ----------
    sghetero_src_nid: int
        id of the source node in the subgraph

    sghetero_tgt_nid: int
        id of the target node in the subgraph

    sghetero : heterogeneous dgl graph
        Union of two k-hop subgraphs

    sghetero_feat_nid: Tensor
        The original `ghetero` node ids of subgraph nodes, for feature identification
    
    '''
    # Extract k-hop subgraph centered at the (src, tgt) pair
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid
    
    if src_ntype == tgt_ntype:
        pred_dict = {src_ntype: torch.tensor([src_nid, tgt_nid])}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype][0]
        sghetero_tgt_nid = inv_dict[tgt_ntype][1]
    else:
        pred_dict = {src_ntype: src_nid, tgt_ntype: tgt_nid}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype]
        sghetero_tgt_nid = inv_dict[tgt_ntype]

    sghetero_feat_nid = sghetero.ndata[dgl.NID]
    
    return sghetero_src_nid, sghetero_tgt_nid, sghetero, sghetero_feat_nid


'''
Path finding utils
'''
def get_neg_path_score_func(g, weight, exclude_node=[],community=None,community_strength=None):
    '''
    Compute the negative path score for the shortest path algorithm.
    
    Parameters
    ----------
    g : dgl graph

    weight: string
       The edge weights stored in g.edata

    exclude_node : iterable
        Degree of these nodes will be set to 0 when computing the path score, so they will likely be included.

    Returns
    ----------
    neg_path_score_func: callable function
       Takes in two node ids and return the edge weight. 
    '''
    log_eweights = g.edata[weight].log().tolist()
    log_in_degrees = g.in_degrees().log()
    log_in_degrees[exclude_node] = 0
    log_in_degrees = log_in_degrees.tolist()
    u, v = g.edges()
    #neg_path_score_map = {edge : log_in_degrees[edge[1]] - log_eweights[i] for i, edge in enumerate(zip(u.tolist(), v.tolist()))}
    
    
     
    neg_path_score_map = {edge : log_in_degrees[edge[1]] - log_eweights[i] - abs( community_strength[edge[0]] - community_strength[edge[1]]) for i, edge in enumerate(zip(u.tolist(), v.tolist()))}



    def neg_path_score_func(u, v):
        return neg_path_score_map[(u, v)]
    return neg_path_score_func

def bidirectional_dijkstra(g, src_nid, tgt_nid, weight=None, ignore_nodes=None, ignore_edges=None):
    """Dijkstra's algorithm for shortest paths using bidirectional search.
    
    Adapted from NetworkX _bidirectional_dijkstra
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html
    
    Parameters
    ----------
    g : dgl graph

    src_nid : int
        source node id

    tgt_nid : int
        target node id

    weight: callable function, optional 
       Takes in two node ids and return the edge weight. 

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    """
    if src_nid == tgt_nid:
        return (0, [src_nid])

    src, tgt = g.edges()
    Gpred = lambda i: src[tgt == i].tolist()
    Gsucc = lambda i: tgt[src == i].tolist()
    
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)
    
    if ignore_edges:
        def filter_pred_iter(pred_iter):
            def iterate(v):
                for w in pred_iter(v):
                    if (w, v) not in ignore_edges:
                        yield w

            return iterate

        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w

            return iterate

        Gpred = filter_pred_iter(Gpred)
        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{src_nid: [src_nid]}, {tgt_nid: [tgt_nid]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{src_nid: 0}, {tgt_nid: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), src_nid))
    push(fringe[1], (0, next(c), tgt_nid))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    if not weight:
        weight = lambda u, v: 1
            
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = weight(v, w)
                vwLength = dists[dir][v] + minweight
            else:  # back, must remember to change v,w->w,v
                minweight = weight(w, v)
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise ValueError("No paths found")


class PathBuffer:
    """For shortest paths finding
    
    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    """
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path
    
def k_shortest_paths_generator(g, 
                               src_nid, 
                               tgt_nid, 
                               weight=None, 
                               k=5, 
                               ignore_nodes_init=None,
                               ignore_edges_init=None,
                               community=None,
                               community_strength=None):
    """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
       each with maximum lenghth `max_length`, return starting from the shortest ones. 
       If a weighted shortest path search is to be used, no negative weights are allowed.

    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    Parameters
    ----------
    g : dgl graph

    src_nid : int
        source node id

    tgt_nid : int
        target node id

    weight: callable function, optional 
       Takes in two node ids and return the edge weight. 

    k: int
       number of paths
    
    ignore_nodes_init : set of nodes
       nodes to ignore, optional

    ignore_edges_init : set of edges
       edges to ignore, optional

    Returns
    -------
    path_generator: generator
       A generator that produces lists of tuples (path score, path), in order from
       shortest to longest. Each path is a list of node ids

    """
    if not weight:
        weight = lambda u, v: 1

    def length_func(path):
        return sum(weight(u, v) for (u, v) in zip(path, path[1:]))

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while not prev_path or len(listA) < k:
        if not prev_path:
            length, path = bidirectional_dijkstra(g, src_nid, tgt_nid, weight, ignore_nodes_init, ignore_edges_init)
            #listB.push(length, path)
            
            com_num =len(set(community[node] for node in path))            
            listB.push(length+1/com_num, path)            
            
            
        else:
            ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
            ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = bidirectional_dijkstra(g,
                                                          root[-1],
                                                          tgt_nid,
                                                          ignore_nodes=ignore_nodes,
                                                          ignore_edges=ignore_edges,
                                                          weight=weight)
                    path = root[:-1] + spur
                    #listB.push(root_length + length, path)
                    
                    com_num = len(set(community[node] for node in path))                    
                    listB.push(root_length + length +1/ com_num, path)                    
                    
                    
                except ValueError:
                    pass
                ignore_nodes.add(root[-1])
        
        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

def k_shortest_paths_with_max_length(g, 
                                     src_nid, 
                                     tgt_nid, 
                                     weight=None, 
                                     k=5, 
                                     max_length=None,
                                     ignore_nodes=None,
                                     ignore_edges=None,
                                     community=None,
                                     community_strength=None):
    
    """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
       each with maximum lenghth `max_length`, return starting from the shortest ones. 
       If a weighted shortest path search is to be used, no negative weights are allowed.
   
    Parameters
    ----------
       See function `k_shortest_paths_generator`
   
    Return
    -------
    paths: list of lists
       Each list is a path containing node ids
    """
    path_generator = k_shortest_paths_generator(g, 
                                                src_nid, 
                                                tgt_nid, 
                                                weight=weight,
                                                k=k, 
                                                ignore_nodes_init=ignore_nodes,
                                                ignore_edges_init=ignore_edges,
                                                community=community,
                                                community_strength=community_strength)
    
    try:
        if max_length:
            paths = [path for path in path_generator if len(path) <= max_length + 1]
        else:
            paths = list(path_generator)

    except ValueError:
        paths = [[]]

    return paths

'''
Evaluation utils
'''
def get_comp_g_edge_labels(comp_g, edge_labels):
    """Turn `edge_labels` with node ids in the original graph to
       `comp_g_edge_labels` with node ids in the computation graph.
       For easier evaluation.

    Parameters
    ----------
    comp_g : heterogeneous dgl graph
        computation graph, with .ndata stores key dgl.NID
    
    edge_labels : dict
        key=edge type, value=(source node ids, target node ids)
   
    Return
    -------
    comp_g_edge_labels: dict
        key=edge type, value=a tensor of labels, each label is in {0, 1}
    """
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_comp_g_max_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        if nids.numel() > 0:
            max_nid = nids.max().item()
        else: 
            max_nid = -1

        ntypes_to_comp_g_max_nids[ntype] = max_nid

        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        # The i-th entry will be the nid in comp_g for the i-th node in g
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids


    comp_g_edge_labels = {}
    for can_etype in edge_labels:
        start_ntype, etype, end_ntype = can_etype
        start_nids, end_nids = edge_labels[can_etype]
        start_comp_g_max_nid, end_comp_g_max_nid = ntypes_to_comp_g_max_nids[start_ntype], ntypes_to_comp_g_max_nids[end_ntype]

        # For edges in label but not in comp_g, exclude them
        start_included_nid_mask = start_nids <= start_comp_g_max_nid
        end_included_nid_mask = end_nids <= end_comp_g_max_nid
        comp_g_included_nid_mask = end_included_nid_mask & start_included_nid_mask

        start_nids = start_nids[comp_g_included_nid_mask]
        end_nids = end_nids[comp_g_included_nid_mask]

        comp_g_start_nids = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nids]
        comp_g_end_nids = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nids]
        comp_g_eids = comp_g.edge_ids(comp_g_start_nids.tolist(), comp_g_end_nids.tolist(), etype=etype)

        num_edges = comp_g.num_edges(etype=can_etype)
        comp_g_eid_mask = torch.zeros(num_edges)
        comp_g_eid_mask[comp_g_eids] = 1

        comp_g_edge_labels[can_etype] = comp_g_eid_mask

    return comp_g_edge_labels    

def get_comp_g_path_labels(comp_g, path_labels):
    """Turn `path_labels` with node ids in the original graph
       `comp_g_path_labels` with node ids in the computation graph
       For easier evaluation.

    Parameters
    ----------
    comp_g : heterogeneous dgl graph
        computation graph, with .ndata stores key dgl.NID
    
    path_labels : list of lists
        Each list is a path, i.e., triples of 
        (cannonical edge type, source node id, target node id)
   
    Returns
    -------
    comp_g_path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)
    """
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_comp_g_max_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        if nids.numel() > 0:
            max_nid = nids.max().item()
        else: 
            max_nid = -1

        ntypes_to_comp_g_max_nids[ntype] = max_nid

        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        # The i-th entry will be the nid in comp_g for the i-th node in g
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids

    comp_g_path_labels = []
    for path in path_labels:
        comp_g_path = []
        for can_etype, start_nid, end_nid in path:
            start_ntype, etype, end_ntype = can_etype

            comp_g_start_nid = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nid].item()
            comp_g_end_nid = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nid].item()

            comp_g_eid = comp_g.edge_ids(comp_g_start_nid, comp_g_end_nid, etype=can_etype)
            comp_g_path += [(can_etype, comp_g_eid)]
        comp_g_path_labels += [comp_g_path]
    return comp_g_path_labels

def eval_edge_mask_auc(edge_mask_dict, edge_labels):
    '''
    Evaluate the AUC of an edge mask
    
    Parameters
    ----------
    edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in (-inf, inf)

    edge_labels: dict
        key=edge type, value=a tensor of labels, each label is in {0, 1}

    Returns
    ----------
    ROC-AUC score : int
    '''
    
    y_true = []
    y_score = []
    for can_etype in edge_labels:
        y_true += [edge_labels[can_etype]]
        y_score += [edge_mask_dict[can_etype].detach().sigmoid()]

    y_true = torch.cat(y_true)
    y_score = torch.cat(y_score)
    
    return roc_auc_score(y_true, y_score) 
    
def eval_edge_mask_auc_intra_inter(edge_mask_dict, edge_labels,comp_g, comp_g_feat_nids,membership):
    '''
    Evaluate the AUC of an edge mask
    
    Parameters
    ----------
    edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in (-inf, inf)

    edge_labels: dict
        key=edge type, value=a tensor of labels, each label is in {0, 1}

    Returns
    ----------
    ROC-AUC score : int
    '''
    
    y_true = []
    y_score = []
    
    y_inter_true=[]
    y_inter_score=[]
    
    y_intro_true=[]
    y_intro_score=[]
    
    #print(111,comp_g)
    
    comp_g_node_info={}
    for node_type in comp_g.ntypes:
      comp_g_node_info[node_type] = comp_g.nodes(node_type)
    #print(222,comp_g_node_info)
    
        
    #print(333,comp_g_feat_nids)
    #print(444,membership)
    
    feat_nids_ori = sorted({(key, int(value)) for key, values in comp_g_feat_nids.items() for value in values.cpu().numpy()})
    mg_membership = {node: membership[node] for node in feat_nids_ori if node in membership}#
    #print(555,feat_nids_ori)
    #print(666,mg_membership)
    com_g_keys=sorted({(key, int(value)) for key, values in comp_g_node_info.items() for value in values.cpu().numpy()})
    #print(777,com_g_keys)
    com_g_values=mg_membership.values()
    #print(888,com_g_values)
    com_g_membership={k: v for k, v in zip(com_g_keys, com_g_values)}
    #print(999,com_g_membership)
    
    
    for can_etype in edge_labels:
        y_true += [edge_labels[can_etype]]
        y_score += [edge_mask_dict[can_etype].detach().sigmoid()]
        
        
        
        #print(22,[edge_labels[can_etype]])
        
        #print(33,len([edge_labels[can_etype]][0]))

              
        #print(1100,can_etype)
        start_ntype, e_etype, end_ntype = can_etype
        u, v = comp_g.edges(etype=e_etype)
        M = u.shape[0] # number of directed edges
        eids = torch.arange(M)
        #print(44,eids )             
        #print(1111,u, v)
        u=u.tolist()
        v=v.tolist()
        edges=[(x, y) for x, y in zip(u, v)]
        #print(2222,edges,len(edges))
                
        
        
        edges_intro_id=[]
        edges_inter_id=[]
        
        
        
        
        for index, (i,j) in enumerate(edges):
            
            #print(3333,index,(start_ntype,i),(end_ntype,j))
            #print(4444,com_g_membership[(start_ntype,i)],com_g_membership[(end_ntype,j)])
            if  com_g_membership[(start_ntype,i)] == com_g_membership[(end_ntype,j)]:
              edges_intro_id.append(index)
            else:
              edges_inter_id.append(index)
                
        #print(555,edges_intro_id)
        #print(666,edges_inter_id)
        
        edges_intro_id=torch.tensor(edges_intro_id, dtype=torch.int64)
        edges_inter_id=torch.tensor(edges_inter_id, dtype=torch.int64)
        y_intro_true += [torch.index_select(edge_labels[can_etype], 0, edges_intro_id)]
        y_inter_true += [torch.index_select(edge_labels[can_etype], 0, edges_inter_id)]
             
              
        y_intro_score += [torch.index_select(edge_mask_dict[can_etype].detach().sigmoid(),0,edges_intro_id)]
        y_inter_score += [torch.index_select(edge_mask_dict[can_etype].detach().sigmoid(),0,edges_inter_id)]
        
        
    y_true = torch.cat(y_true)
    y_score = torch.cat(y_score)
    #y_score_mean = y_score.mean().item()
    
    if y_score.numel() > 0:
      y_score_mean = y_score.mean().item()
    else:
      y_score_mean = 0
        
    y_intro_true = torch.cat(y_intro_true)
    y_intro_score = torch.cat(y_intro_score)
    #y_intro_score_mean = y_intro_score.mean().item()
    
    if y_intro_score.numel() > 0:
      y_intro_score_mean = y_intro_score.mean().item()
      y_intro_score_all = y_intro_score.sum().item()
    else:
      y_intro_score_mean = 0
      y_intro_score_all =0  
      
          
    
        
    y_intro_pred = (y_intro_score >= 0.5).to(torch.int32)
    y_intro_positive = torch.sum(y_intro_pred==1).item()
    y_intro_negitive = torch.sum(y_intro_pred==0).item()
    y_intro_all =y_intro_pred.numel() 
    y_intro_true_positive = torch.sum(y_intro_true==1).item() 
    y_intro_true_negetive = torch.sum(y_intro_true==0).item()  
           
    y_inter_score = torch.cat(y_inter_score)
    y_inter_true = torch.cat(y_inter_true)
    y_inter_pred = (y_inter_score >= 0.5).to(torch.int32)
    y_inter_positive = torch.sum(y_inter_pred==1).item()
    y_inter_negitive = torch.sum(y_inter_pred==0).item()
    y_inter_all =y_inter_pred.numel() 
    y_inter_true_positive = torch.sum(y_inter_true==1).item() 
    y_inter_true_negetive = torch.sum(y_inter_true==0).item()         
    #y_inter_score_mean = y_inter_score.mean().item()
    
    if y_inter_score.numel() > 0:
      y_inter_score_mean = y_inter_score.mean().item()
      y_inter_score_all = y_inter_score.sum().item()
    else:
      y_inter_score_mean = 0
      y_inter_score_all = 0     
    
    #print(11111,y_inter_positive,y_intro_positive,y_inter_score_mean,y_intro_score_mean)
    
    Disparate_Impact=y_inter_positive/y_intro_positive if y_intro_positive != 0 else 0
    Disparate_Impact_probability=y_inter_score_mean/y_intro_score_mean if y_intro_score_mean != 0 else 0
    
    Demographic_parity= abs(y_inter_score_mean-y_intro_score_mean)
    #Demographic_parity= abs(y_inter_score_all-y_intro_score_all)
    
    BER=(y_inter_positive-y_intro_positive+1)/2
    BER_probability=(y_inter_score_mean-y_intro_score_mean+1)/2
    
                       
    sp_intro=y_intro_positive/y_intro_all if y_intro_all != 0 else 0         
    sp_inter=y_inter_positive/y_inter_all if y_inter_all != 0 else 0
    sp=abs(sp_intro-sp_inter) 
    


    y_intro_positive_index=(y_intro_pred == 1).nonzero(as_tuple=False)
    y_intro_true_positive_index = (y_intro_true == 1).nonzero(as_tuple=False)   
    y_intro_negetive_index=(y_intro_pred == 0).nonzero(as_tuple=False)
    y_intro_true_negetive_index = (y_intro_true == 0).nonzero(as_tuple=False)        
    y_intro_true_pred_positive_value = torch.isin(y_intro_positive_index,y_intro_true_positive_index)    
    y_intro_true_pred_positive_index = y_intro_positive_index[y_intro_true_pred_positive_value]
    y_intro_true_pred_positive_common_values=y_intro_score[y_intro_true_pred_positive_index]
    y_intro_true_pred_positive_common_values_sum=y_intro_true_pred_positive_common_values.sum()        
    y_intro_true_pred_positive = y_intro_true_pred_positive_value.sum().item()
    
    
    y_inter_positive_index=(y_inter_pred == 1).nonzero(as_tuple=False)
    y_inter_true_positive_index = (y_inter_true == 1).nonzero(as_tuple=False)    
    y_inter_negetive_index=(y_inter_pred == 0).nonzero(as_tuple=False)
    y_inter_true_negetive_index = (y_inter_true == 0).nonzero(as_tuple=False)        
    y_inter_true_pred_positive_value = torch.isin(y_inter_positive_index,y_inter_true_positive_index)    
    y_inter_true_pred_positive_index = y_inter_positive_index[y_inter_true_pred_positive_value]
    y_inter_true_pred_positive_common_values=y_inter_score[y_inter_true_pred_positive_index]
    y_inter_true_pred_positive_common_values_sum=y_inter_true_pred_positive_common_values.sum()        
    y_inter_true_pred_positive = y_inter_true_pred_positive_value.sum().item()    
    


    
    
    
    y_intro_true_negetive_pred_positivev_value = torch.isin(y_intro_true_negetive_index, y_intro_positive_index)   
    y_intro_true_negetive_pred_positive = y_intro_true_negetive_pred_positivev_value.sum().item()
    
    y_inter_true_negetive_pred_positivev_value = torch.isin(y_inter_true_negetive_index, y_inter_positive_index)   
    y_inter_true_negetive_pred_positive = y_inter_true_negetive_pred_positivev_value.sum().item()    
   
    
        

    Equal_opportunity_intro =y_intro_true_pred_positive/y_intro_true_positive if y_intro_true_positive != 0 else 0
    Equal_opportunity_intro_pro =y_intro_true_pred_positive_common_values_sum/y_intro_true_positive if y_intro_true_positive != 0 else 0
    
    Equal_opportunity_inter=y_inter_true_pred_positive/y_inter_true_positive if y_inter_true_positive != 0 else 0
    Equal_opportunity_inter_pro=y_inter_true_pred_positive_common_values_sum/y_inter_true_positive if y_inter_true_positive != 0 else 0
    #print(l)
    
    FPR_intro=y_intro_true_negetive_pred_positive/y_intro_true_negetive if y_intro_true_negetive != 0 else 0
    FPR_inter=y_inter_true_negetive_pred_positive/y_inter_true_negetive if y_inter_true_negetive != 0 else 0
    
    Equalized_Odds_FPR=abs(FPR_intro-FPR_inter)
    Equalized_Odds_TPR=abs(Equal_opportunity_intro-Equal_opportunity_inter)
     
  
    
    
    
    
    Equal_opportunity=abs(Equal_opportunity_intro-Equal_opportunity_inter)
    Equal_opportunity_pro=abs(Equal_opportunity_intro_pro-Equal_opportunity_inter_pro)    
    
    Equalized_Odds=max(Equalized_Odds_FPR,Equalized_Odds_TPR)
    
        
       
   
        
    all_auc =roc_auc_score(y_true, y_score)

    if torch.any(y_intro_true.eq(1)) and torch.any(y_inter_true.eq(1)): 
      inter_auc =   roc_auc_score(y_inter_true, y_inter_score)
      intro_auc =   roc_auc_score(y_intro_true, y_intro_score)    
    else:
      inter_auc =0.5
      intro_auc = 0.5 

       
    
  
        
      

      
    return all_auc,intro_auc,inter_auc, y_score_mean,y_intro_score_mean,y_inter_score_mean,sp,Demographic_parity,Equal_opportunity,Equal_opportunity_pro,Equalized_Odds_FPR,Equalized_Odds_TPR,Equalized_Odds,Disparate_Impact,Disparate_Impact_probability,BER,BER_probability

    
      
      
    

      
    
    





def eval_edge_mask_topk_path_hit(edge_mask_dict, path_labels, topks=[10]):
    '''
    Evaluate the path hit rate of the top k edges in an edge mask
    
    Parameters
    ----------
    edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in (-inf, inf)

    path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)

    topks: iterable
        An iterable of the top `k` values. Each `k` determines how many edges to select 
        from the top values of the mask.

    Returns
    ----------
    topk_to_path_hit: dict
        Mapping the top `k` to 
    '''
    cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
    M = len(cat_edge_mask)
    topks = {k: min(k, M) for k in topks}

    topk_to_path_hit = defaultdict(list)
    for r, k in topks.items():
        threshold = cat_edge_mask.topk(k)[0][-1].item()
        hard_edge_mask_dict = {}
        for etype in edge_mask_dict:
            hard_edge_mask_dict[etype] = edge_mask_dict[etype] >= threshold

        hit = eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels)
        topk_to_path_hit[r] += [hit]
    return topk_to_path_hit

def eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels):
    '''
    Evaluate the path hit of the an hard edge mask
    
    Parameters
    ----------
    hard_edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in {True, False}

    path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)

    Returns
    ----------
    hit_path: int
        1 or 0
    '''
    for path in path_labels:
        hit_path = 1
        for can_etype, eid in path:
            if not hard_edge_mask_dict[can_etype][eid]:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0


def eval_path_explanation_edges_path_hit(path_explanation_edges, path_labels):
    '''
    Evaluate the path hit rate of the a path_explanation_edges
    
    Parameters
    ----------
    path_explanation_edges : list
        Edges on the path explanation, each edge is a triples of 
        (cannonical edge type, source node id, target node id)
    
    path_labels : list of lists
        Each list is a path, i.e., triples of 
        (cannonical edge type, source node id, target node id)

    Returns
    ----------
    hit_path: int
        1 or 0
    '''
    for path in path_labels:
        hit_path = 1
        for edge in path:
            if edge not in path_explanation_edges:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0


'''
Plotting utils
'''
def plot_hetero_graph(ghetero,
                      ntypes_to_nshapes=None,
                      ntypes_to_ncolors=None,
                      ntypes_to_nlayers=None,
                      layout='multipartite',
                      layout_seed=0,
                      node_size=1000,
                      edge_kwargs={},
                      selected_node_dict=None,
                      selected_node_color='red',
                      selected_edge_dict=None,
                      selected_edge_kwargs={},
                      label='nid',
                      etype_label=True,
                      label_offset=False,
                      title=None,
                      legend=False,
                      figsize=(10, 10),
                      fig_name=None,
                      fig_format='png',
                      is_show=True):
        '''
        Parameters
        ----------
        ghetero: a DGL heterogeneous graph with ndata `order`

        ntypes_to_nshapes : Dict
            mapping node types to node shapes
        
        ntypes_to_ncolors : Dict
            mapping node types to node colors

        ntypes_to_nlayers : Dict 
            mapping node types to layer order in the multipartite layout. 

        label: String
            one of ['none', nid'] or a node feature stored in ndata of ghetero

        Returns
        ----------
        nx_graph : networkx graph
        
        '''
        if ntypes_to_nshapes is None:
            default_node_shape = 'o'
        if ntypes_to_ncolors is None:
            default_node_color = 'cyan'
        if selected_node_dict is not None:
            selected_node_dict = {ntype: list(selected_node_dict[ntype]) for ntype in selected_node_dict}

        # Convert DGL graph to networkx graph
        ghomo = dgl.to_homogeneous(ghetero)
        edges = torch.cat([t.unsqueeze(1) for t in ghomo.edges()], dim=1)
        edge_list = [(n_frm, n_to) for (n_frm, n_to) in edges.tolist()]
        nx_graph = dgl.to_networkx(ghomo, node_attrs=[dgl.NTYPE])
            
        # Use different layout
        if layout == 'spring':
            pos = nx.spring_layout(nx_graph, seed=layout_seed)
        elif layout == 'kk':
            pos = nx.kamada_kawai_layout(nx_graph)
        elif layout == 'multipartite':
            if ntypes_to_nlayers is not None:
                ntype_ids_to_nlayers = {ghetero.get_ntype_id(ntype): ntypes_to_nlayers[ntype] for ntype in ghetero.ntypes}
            else:
                ntype_ids_to_nlayers = {ghetero.get_ntype_id(ntype): i for i, ntype in enumerate(ghetero.ntypes)}
                
            for i in nx_graph.nodes():
                ntype_id = nx_graph.nodes()[i][dgl.NTYPE].item()
                nx_graph.nodes()[i][dgl.NTYPE] = ntype_ids_to_nlayers[ntype_id]

            pos = nx.multipartite_layout(nx_graph, subset_key=dgl.ETYPE, scale=1)
        else:
            raise ValueError('Unknown layout')

        # Start drawing
        plt.figure(figsize=figsize)
        ax = plt.gca()
 
        # Draw nodes for each ntype
        for ntype in ghetero.ntypes:
            ntype_ids = ghomo.ndata[dgl.NTYPE]
            hetero_nids = ghomo.ndata[dgl.NID] # nid in the original hetero graph
            
            node_shape = ntypes_to_nshapes[ntype] if ntypes_to_nshapes else default_node_shape
            node_color = ntypes_to_ncolors[ntype] if ntypes_to_ncolors else default_node_color

            # For the current node type, get the node type id and node ids
            curr_ntype_id = ghetero.get_ntype_id(ntype)
            curr_nids_mask = ntype_ids == curr_ntype_id
            curr_nids = curr_nids_mask.nonzero().view(-1).tolist()

            # For the current node type, get node ids and prediction node id in the original hetero graph
            curr_hetero_nids = hetero_nids[curr_nids_mask]
            
            if selected_node_dict is not None:
                curr_hetero_selected_nid = selected_node_dict.get(ntype)
                if curr_hetero_selected_nid is not None:
                    curr_node_color = []
                    for hetero_nid in curr_hetero_nids:
                        curr_node_color += [selected_node_color if hetero_nid in curr_hetero_selected_nid else node_color]
                    node_color = curr_node_color

            nx.draw_networkx_nodes(nx_graph, 
                                   pos, 
                                   curr_nids, 
                                   node_shape=node_shape,
                                   node_color=node_color,
                                   node_size=node_size,
                                   ax=ax)
            
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, edge_list, **edge_kwargs, ax=ax)
        
        if selected_edge_dict is not None:
            ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)
            homo_selected_edge_list = []
            for etype in selected_edge_dict:
                src_ntype, _, tgt_ntype = ghetero.to_canonical_etype(etype)
                src_nids, tgt_nids = selected_edge_dict[etype]
                for src_nid, tgt_nid in zip(src_nids.tolist(), tgt_nids.tolist()):
                    homo_src_nid = ntype_hetero_nids_to_homo_nids[(src_ntype, src_nid)]
                    homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(tgt_ntype, tgt_nid)]
                    homo_selected_edge_list += [(homo_src_nid, homo_tgt_nid)]
        
            nx.draw_networkx_edges(nx_graph, pos, homo_selected_edge_list, **selected_edge_kwargs, ax=ax)
            
            
     # Start labelling nodes
        if label == 'none':
            pass
        elif label == 'nid':
            homo_nids_to_hetero_nids = get_homo_nids_to_hetero_nids(ghetero)
            nx.draw_networkx_labels(nx_graph, pos, labels=homo_nids_to_hetero_nids)
        else:
            # Set extra space to avoid label outside of the box
            x_values, y_values = zip(*pos.values())
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * 0.12
            ax.set_xlim(x_min - x_margin, x_max + x_margin)


            if ghetero.ndata.get(label):
                homo_nids_to_hetero_ndata_feat = get_homo_nids_to_hetero_ntype_data_feat(ghetero, label)
                if label_offset:
                    offset = 0.8 / figsize[1]
                    label_pos = {nid : [p[0], p[1] - offset] for nid, p in pos.items()} 
                else:
                    label_pos = pos

                nx.draw_networkx_labels(nx_graph, 
                                        label_pos, 
                                        font_size=14, 
                                        font_weight='bold', 
                                        labels=homo_nids_to_hetero_ndata_feat,
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        ax=ax)

            else:
                raise ValueError('Unrecognized label')
            
        # Start labelling edges with etype
        if etype_label is not None:
            if ghetero.ndata.get(label):
                homo_nid_pairs_to_etypes = get_homo_nid_pairs_to_etypes(ghetero)
                nx.draw_networkx_edge_labels(nx_graph, 
                                             pos, 
                                             font_size=13, 
                                             font_weight='bold', 
                                             edge_labels=homo_nid_pairs_to_etypes,
                                             horizontalalignment='center',
                                             verticalalignment='center',
                                             ax=ax)
            
        if legend:
            plt.legend(ghetero.ntypes, fontsize=15, prop={'size': figsize[0]*2.5}, bbox_to_anchor = (1.15, 0.7)) 

        ax.axis('off')
        if title is not None:
            plt.title(textwrap.fill(title, width=60))
        if fig_name is not None:
            plt.savefig(fig_name, format=fig_format, bbox_inches='tight')
        if is_show:
            plt.show()
        if fig_name is not None:
            plt.close()
            
        return nx_graph
  
def get_homo_nids_to_hetero_ntype_data_feat(ghetero, feat=dgl.NID):
    '''
    Plotting helper function
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_ndata_feat = []
    for ntype in ghetero.ntypes:
        hetero_ndata_feat += [f'{ntype[0]}' + f'{feat}' for feat in ghetero.ndata[feat][ntype].tolist()]

    homo_nids_to_hetero_ndata_feat = dict(zip(homo_nids, hetero_ndata_feat))
    return homo_nids_to_hetero_ndata_feat

def get_homo_nid_pairs_to_etypes(ghetero):
    '''
    Plotting helper function
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    etypes = ghetero.etypes
    etype_list = [etypes[etype_id] for etype_id in ghomo.edata[dgl.ETYPE]]
    u, v = ghomo.edges()
    homo_nid_pairs_to_etypes = dict(zip(zip(u.tolist(), v.tolist()), etype_list))
    return homo_nid_pairs_to_etypes


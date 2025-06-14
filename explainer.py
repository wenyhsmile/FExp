import dgl
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from collections import Counter
from collections import defaultdict
from utils import get_ntype_hetero_nids_to_homo_nids, get_homo_nids_to_ntype_hetero_nids, get_ntype_pairs_to_cannonical_etypes
from utils import hetero_src_tgt_khop_in_subgraph, get_neg_path_score_func, k_shortest_paths_with_max_length,shifted_sigmoid,louvain_partition

def get_edge_mask_dict(ghetero):
    '''
    Create a dictionary mapping etypes to learnable edge masks 
            
    Parameters
    ----------
    ghetero : heterogeneous dgl graph.

    Return
    ----------
    edge_mask_dict : dictionary
        key=etype, value=torch.nn.Parameter with size number of etype edges
    '''
    device = ghetero.device
    edge_mask_dict = {}
    default_std = 1.0
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()
        if num_nodes == 0:
            std = default_std
        else:
            std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)
    return edge_mask_dict
    
    
def get_edge_mask_dict22(ghetero):
    '''
    Create a dictionary mapping etypes to learnable edge masks 
            
    Parameters
    ----------
    ghetero : heterogeneous dgl graph.

    Return
    ----------
    edge_mask_dict : dictionary
        key=etype, value=torch.nn.Parameter with size number of etype edges
    '''
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)
    return edge_mask_dict

def remove_edges_of_high_degree_nodes(ghomo, max_degree=10, always_preserve=[]):
    '''
    For all the nodes with degree higher than `max_degree`, 
    except nodes in `always_preserve`, remove their edges. 
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    max_degree : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    low_degree_ghomo : dgl homogeneous graph
        Pruned graph with edges of high degree nodes removed

    '''
    d = ghomo.in_degrees()
    high_degree_mask = d > max_degree
    
    # preserve nodes
    high_degree_mask[always_preserve] = False    

    high_degree_nids = ghomo.nodes()[high_degree_mask]
    u, v = ghomo.edges()
    high_degree_edge_mask = torch.isin(u, high_degree_nids) | torch.isin(v, high_degree_nids)
    high_degree_u, high_degree_v = u[high_degree_edge_mask], v[high_degree_edge_mask]
    high_degree_eids = ghomo.edge_ids(high_degree_u, high_degree_v)
    low_degree_ghomo = dgl.remove_edges(ghomo, high_degree_eids)
    
    return low_degree_ghomo





    

    
    
def remove_edges_except_k_core_graph(ghomo, k, always_preserve=[],community_strength=None,community=None):
    '''
    Remove edges of low importance nodes based on PageRank algorithm, while keeping the nodes intact.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    threshold : float
        The threshold for removing edges based on PageRank value of nodes.
    
    always_preserve : iterable
        Nodes in this list won't have their edges removed.
    
    Returns
    -------
    pruned_ghomo : dgl homogeneous graph
        The graph with edges removed based on PageRank importance
    '''
    k_core_ghomo = ghomo
    
    
    g_nx = dgl.to_networkx(k_core_ghomo.cpu(), node_attrs=[dgl.NTYPE]).to_undirected()
    node_betweenness = nx.degree_centrality(g_nx)
    degrees = torch.tensor(list(node_betweenness.values()))

    k=0.0005
    
    
    k_core_mask = (degrees > 0) & (degrees < k)
    
    k_core_mask[always_preserve] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)

        g_nx = dgl.to_networkx(k_core_ghomo.cpu(), node_attrs=[dgl.NTYPE]).to_undirected()
        node_betweenness = nx.degree_centrality(g_nx)
        degrees = torch.tensor(list(node_betweenness.values()))

        
        
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False

    return k_core_ghomo
    
    

    
    
def remove_edges_except_k_core_graph22(ghomo, k, always_preserve=[]):
    '''
    Find the `k`-core of `ghomo`.
    Only isolate the low degree nodes by removing theirs edges
    instead of removing the nodes, so node ids can be kept.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    k : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    k_core_ghomo : dgl homogeneous graph
        The k-core graph
    '''
    k_core_ghomo = ghomo
    degrees = k_core_ghomo.in_degrees()
    k_core_mask = (degrees > 0) & (degrees < k)
    k_core_mask[always_preserve] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)
        
        degrees = k_core_ghomo.in_degrees()
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False

    return k_core_ghomo

def get_eids_on_paths(paths, ghomo):
    '''
    Collect all edge ids on the paths
    
    Note: The current version is a list version. An edge may be collected multiple times
    A different version is a set version where an edge can only contribute one time 
    even it appears in multiple paths
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    Returns
    -------
    paths: list of lists
        Each list contains (source node ids, target node ids)
        
    '''
    row, col = ghomo.edges()
    eids = []
    for path in paths:
        for i in range(len(path)-1):
            eids += [((row == path[i]) & (col == path[i+1])).nonzero().item()]
            
    return torch.LongTensor(eids)

def comp_g_paths_to_paths(comp_g, comp_g_paths):
    paths = []
    g_nids = comp_g.ndata[dgl.NID]
    for comp_g_path in comp_g_paths:
        path = []
        for can_etype, u, v in comp_g_path:
            u_ntype, _, v_ntype = can_etype
            path += [(can_etype, g_nids[u_ntype][u].item(), g_nids[v_ntype][v].item())]
        paths += [path]
    return paths


class PaGELink(nn.Module):
    """Path-based GNN Explanation for Heterogeneous Link Prediction (PaGELink)
    
    Some methods are adapted from the DGL GNNExplainer implementation
    https://docs.dgl.ai/en/0.8.x/_modules/dgl/nn/pytorch/explain/gnnexplainer.html#GNNExplainer
    
    Parameters
    ----------
    model : nn.Module
        The GNN-based link prediction model to explain.

        * The required arguments of its forward function are source node id, target node id,
          graph, and feature ids. The feature ids are for selecting input node features.
        * It should also optionally take an eweight argument for edge weights
          and multiply the messages by the weights during message passing.
        * The output of its forward function is the logits in (-inf, inf) for the 
          predicted link.
    lr : float, optional
        The learning rate to use, default to 0.01.
    num_epochs : int, optional
        The number of epochs to train.
    alpha1 : float, optional
        A higher value will make the explanation edge masks more sparse by decreasing
        the sum of the edge mask.
    alpha2 : float, optional
        A higher value will make the explanation edge masks more discrete by decreasing
        the entropy of the edge mask.
    alpha : float, optional
        A higher value will make edges on high-quality paths to have higher weights
    beta : float, optional
        A higher value will make edges off high-quality paths to have lower weights
    log : bool, optional
        If True, it will log the computation process, default to True.
    """
    def __init__(self,
                 model,
                 lr=0.001,
                 num_epochs=100,
                 alpha=1.0,
                 beta=1.0,
                 log=False):
        super(PaGELink, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.beta = beta
        self.log = log
        
        self.all_loss = defaultdict(list)

    def _init_masks(self, ghetero):
        """Initialize the learnable edge mask.

        Parameters
        ----------
        graph : DGLGraph
            Input graph.

        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        return get_edge_mask_dict(ghetero)
    

    def _prune_graph(self, ghetero, prune_max_degree=-1, k_core=2, always_preserve=[],community_strength=None,community=None):
        # Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
        # The pruning is computed on the homogeneous graph, i.e., ignoring node/edge types
        ghomo = dgl.to_homogeneous(ghetero)
        device = ghetero.device
        ghomo.edata['eid_before_prune'] = torch.arange(ghomo.num_edges()).to(device)
        
        if prune_max_degree > 0:
            max_degree_pruned_ghomo = remove_edges_of_high_degree_nodes(ghomo, prune_max_degree, always_preserve)
            k_core_ghomo = remove_edges_except_k_core_graph(max_degree_pruned_ghomo, k_core, always_preserve,community_strength,community)
            
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = max_degree_pruned_ghomo
            else:
                pruned_ghomo = k_core_ghomo
        else:
            k_core_ghomo = remove_edges_except_k_core_graph(ghomo, k_core, always_preserve,community_strength,community)
            if k_core_ghomo.num_edges() <= 0: # no k-core found
                pruned_ghomo = ghomo
            else:
                pruned_ghomo = k_core_ghomo
        
        pruned_ghomo_eids = pruned_ghomo.edata['eid_before_prune']
        pruned_ghomo_eid_mask = torch.zeros(ghomo.num_edges()).bool()
        pruned_ghomo_eid_mask[pruned_ghomo_eids] = True

        # Apply the pruning result on the heterogeneous graph
        etypes_to_pruned_ghetero_eid_masks = {}
        pruned_ghetero = ghetero
        cum_num_edges = 0
        for etype in ghetero.canonical_etypes:
            num_edges = ghetero.num_edges(etype=etype)
            pruned_ghetero_eid_mask = pruned_ghomo_eid_mask[cum_num_edges:cum_num_edges+num_edges]
            etypes_to_pruned_ghetero_eid_masks[etype] = pruned_ghetero_eid_mask

            remove_ghetero_eids = (~ pruned_ghetero_eid_mask).nonzero().view(-1).to(device)
            pruned_ghetero = dgl.remove_edges(pruned_ghetero, eids=remove_ghetero_eids, etype=etype)

            cum_num_edges += num_edges
                
        return pruned_ghetero, etypes_to_pruned_ghetero_eid_masks
        
        
    def path_loss(self, src_nid, tgt_nid, g, eweights, pred,communities=None,community_strengths=None,num_paths=2):
        """Compute the path loss.

        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        g : dgl graph

        eweights : Tensor
            Edge weights with shape equals the number of edges.
            
        num_paths : int
            Number of paths to compute path loss on

        Returns
        -------
        loss : Tensor
            The path loss
        """
        neg_path_score_func = get_neg_path_score_func(g, 'eweight', [src_nid, tgt_nid],community=communities,community_strength=community_strengths)
        paths = k_shortest_paths_with_max_length(g, 
                                                 src_nid, 
                                                 tgt_nid, 
                                                 weight=neg_path_score_func, 
                                                 k=num_paths,
                                                 community=communities,
                                                 community_strength=community_strengths)
        
        eids_on_path = get_eids_on_paths(paths, g)

        if eids_on_path.nelement() > 0:
            loss_on_path = - eweights[eids_on_path].mean()
            #loss_on_path = - shifted_sigmoid(eweights[eids_on_path],-6).mean()#10#11
        else:
            loss_on_path = 0

        eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
        
        if eids_off_path_mask.any():
            loss_off_path = eweights[eids_off_path_mask].mean()
            
        else:
            loss_off_path = 0
        loss_differ = max(0,loss_off_path+0.4+loss_on_path)
        
        loss = self.alpha * loss_on_path + self.beta * loss_off_path + self.beta *loss_differ             
        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]

        return loss             
            
            
            
        '''
        eids_on_path = get_eids_on_paths(paths, g)

        if eids_on_path.nelement() > 0:
            loss_on_path =  eweights[eids_on_path].mean()
        else:
            loss_on_path = 0

        eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
        if eids_off_path_mask.any():
            loss_off_path = eweights[eids_off_path_mask].mean()
        else:
            loss_off_path = 0        
        
        softmax= torch.nn.Softmax(dim=-1) 
    
        
        uniform_dis = torch.ones_like(loss_off_path) * (1/2) 
              
        path_loss_mask = (-1) ** pred * shifted_sigmoid( loss_on_path,0).log()
        path_loss_random = -(uniform_dis) ** pred *  shifted_sigmoid( softmax(loss_off_path),0).log()
        

         
        loss =path_loss_mask+path_loss_random#4
        
        
        mse_1 = torch.sum((loss_on_path-pred)**2)
        mse_2 = torch.sum((softmax(loss_off_path)-uniform_dis)**2) 
        loss = self.alpha * mse_1 + self.beta * mse_2 #5#7
              
               
        

        self.all_loss['loss_on_path'] += [float(loss_on_path)]
        self.all_loss['loss_off_path'] += [float(loss_off_path)]

        return loss   
        '''
    
    def get_edge_mask(self, 
                      src_nid, 
                      tgt_nid, 
                      ghetero, 
                      feat_nids,
                      membership,
                      node_community_strengths, 
                      prune_max_degree=-1,
                      k_core=2, 
                      prune_graph=True,
                      with_path_loss=True):

        """Learning the edge mask dict.   
        
        Parameters
        ----------
        see the `explain` method.
        
        Returns
        -------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """

        self.model.eval()
        device = ghetero.device
        
        
        #print(membership)
        
        feat_nids_ori = sorted({(key, int(value)) for key, values in feat_nids.items() for value in values})
        mg_membership = {node: membership[node] for node in feat_nids_ori if node in membership}#
        mg_node_community_strengths = {node: node_community_strengths[node] for node in feat_nids_ori if node in node_community_strengths}#
        mg_membership_values = list(mg_membership.values())
        mg_node_community_strengths_values = list(mg_node_community_strengths.values())
        
        
        
        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]
        
        
        
        
        ntype_hetero_nids_to_homo_nids_value=list(ntype_hetero_nids_to_homo_nids.values())
        ghomo_membership = dict(zip(ntype_hetero_nids_to_homo_nids_value, mg_membership_values))
        ghomo_node_community_strengths = dict(zip(ntype_hetero_nids_to_homo_nids_value, mg_node_community_strengths_values))        
        #print(1111110,ghomo_node_community_strengths)
        

        # Get the initial prediction.
        with torch.no_grad():
            score = self.model(src_nid, tgt_nid, ghetero, feat_nids)
            ori_pred = score.sigmoid()
            #print(ori_pred)
            pred = (score > 0).int().item()
        #print(2222221) 
        if prune_graph:
            # The pruned graph for mask learning  
            ml_ghetero, etypes_to_pruned_ghetero_eid_masks = self._prune_graph(ghetero, 
                                                                               prune_max_degree,
                                                                               k_core,
                                                                            [homo_src_nid, homo_tgt_nid],
                                                                            ghomo_node_community_strengths,ghomo_membership)
        else:
            # The original graph for mask learning  
            ml_ghetero = ghetero
        
        
       
          
    

        
           
            
        ml_edge_mask_dict = self._init_masks(ml_ghetero)
        optimizer = torch.optim.Adam(ml_edge_mask_dict.values(), lr=self.lr)
        
        if self.log:
            pbar = tqdm(total=self.num_epochs)

        eweight_norm = 0
        EPS = 1e-3
        for e in range(self.num_epochs):    
            
            # Apply sigmoid to edge_mask to get eweight
            #ml_eweight_dict_orii = {etype:  ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}
            #print(110,ml_eweight_dict_orii)
            #ml_eweight_dict_random = {etype: 1-ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}
            
            
            
            ml_eweight_dict_ori2 = {etype: ml_edge_mask_dict[etype].sigmoid()   for etype in ml_edge_mask_dict}
            ml_ghetero.edata['eweight'] = ml_eweight_dict_ori2
            
            ml_eweight_dict={}
           
            edges_strenth={}            
            for etype in ml_edge_mask_dict:
              
              
              ml_eweight_dict[etype] = ml_eweight_dict_ori2[etype]                                    
              edges_strenth[etype] = torch.Tensor(ml_eweight_dict[etype].size()[0]).fill_(0).cuda()                             
              src_nodeids, tgt_nodeids = ml_ghetero.edges(etype=etype)                            
              src_nodetype, _, tgt_nodetype = ml_ghetero.to_canonical_etype(etype) 
              
                
              for i, (src_nodeid, tgt_nodeid) in enumerate(zip(src_nodeids.tolist(), tgt_nodeids.tolist())):
                  homo_src_nodeid = ntype_hetero_nids_to_homo_nids[(src_nodetype, src_nodeid)]
                  homo_tgt_nodeid = ntype_hetero_nids_to_homo_nids[(tgt_nodetype, tgt_nodeid)]
                  edges_strenth[etype][i] =  abs(ghomo_node_community_strengths[homo_src_nodeid]  - ghomo_node_community_strengths[homo_tgt_nodeid])*0.5
                                                  
               
              ml_eweight_dict[etype] = ml_eweight_dict[etype] + edges_strenth[etype]
              
            
            ml_eweight_dict_ori = {etype:  ml_eweight_dict[etype] for etype in ml_eweight_dict}
            ml_eweight_dict_random = {etype: 1-ml_eweight_dict[etype] for etype in ml_eweight_dict}
            #print(111,ml_eweight_dict_ori) 
            #print(222,ml_eweight_dict_random)
            '''
            ml_eweight_dict_ori_values = list(ml_eweight_dict_ori.values())
            ml_eweight_dict_ori_tensor = torch.cat(ml_eweight_dict_ori_values, dim=0) 
            
            ml_eweight_dict_random_values = list(ml_eweight_dict_random.values())
            ml_eweight_dict_random_tensor = torch.cat(ml_eweight_dict_random_values, dim=0)           
            
            loss_on_mask=-ml_eweight_dict_ori_tensor.mean()
            loss_off_mask=ml_eweight_dict_random_tensor.mean()

            #print(l)
            
            loss_differ = max(0,loss_off_mask+0.0+loss_on_mask)                  
            pred_loss=loss_on_mask+loss_off_mask+loss_differ#1
            '''
            
            
            
            
            
           
            
            softmax= torch.nn.Softmax(dim=-1) 
           
            score_mask = self.model(src_nid, tgt_nid, ml_ghetero, feat_nids, ml_eweight_dict_ori)
            score_mask = shifted_sigmoid( score_mask,6)
            #print(score_mask)
           
            score_random_mask = self.model(src_nid, tgt_nid, ml_ghetero, feat_nids, ml_eweight_dict_random)
            score_random_mask =shifted_sigmoid( score_random_mask,6)
            #print(score_random_mask)
            #print(l)
            uniform_dis = torch.ones_like(score_random_mask) * (1/2)
            '''        
            pred_loss_mask = (-1) ** pred * shifted_sigmoid( score_mask,6).log()
            pred_loss_random = -(uniform_dis) ** pred *  shifted_sigmoid( softmax(score_random_mask),6).log()
            pred_loss=pred_loss_mask+pred_loss_random#2#
            '''
            
            
            
        
            

            
            
            pmse_1 = torch.sum((score_mask-pred)**2)
            pmse_2 = torch.sum((softmax(score_random_mask)-uniform_dis)**2)
                        
            pred_loss = pmse_1 + pmse_2#3#6#8#9
            
            '''
            score = self.model(src_nid, tgt_nid, ml_ghetero, feat_nids, ml_eweight_dict)
            
            pred_loss =  (-1) ** pred * shifted_sigmoid( score,6).log() 
            '''
            
            self.all_loss['pred_loss'] += [pred_loss.item()]

            ml_ghetero.edata['eweight'] = ml_eweight_dict
            ml_ghomo = dgl.to_homogeneous(ml_ghetero, edata=['eweight'])
            ml_ghomo_eweights = ml_ghomo.edata['eweight']
            
            # Check for early stop
            curr_eweight_norm = ml_ghomo_eweights.norm()
            if abs(eweight_norm - curr_eweight_norm) < EPS:
                break
            eweight_norm = curr_eweight_norm
            
            # Update with path loss
            
            if with_path_loss:
                path_loss = self.path_loss(homo_src_nid, homo_tgt_nid, ml_ghomo, ml_ghomo_eweights,pred,communities=ghomo_membership,community_strengths=ghomo_node_community_strengths)
            else: 
                path_loss = 0
            
            loss = pred_loss + path_loss
            
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.all_loss['total_loss'] += [loss.item()]

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        edge_mask_dict_placeholder = self._init_masks(ghetero)
        edge_mask_dict = {}
        
        if prune_graph:
            # remove pruned edges
            for etype in ghetero.canonical_etypes:
                edge_mask = edge_mask_dict_placeholder[etype].data + float('-inf')    
                pruned_ghetero_eid_mask = etypes_to_pruned_ghetero_eid_masks[etype]
                edge_mask[pruned_ghetero_eid_mask] = ml_edge_mask_dict[etype]
                edge_mask_dict[etype] = edge_mask
                
        else:
            edge_mask_dict = ml_edge_mask_dict
    
        edge_mask_dict = {k : v.detach() for k, v in edge_mask_dict.items()}
        return edge_mask_dict,ghomo_membership,ghomo_node_community_strengths   

    def get_paths22(self,
                  src_nid, 
                  tgt_nid, 
                  ghetero,
                  edge_mask_dict,
                  num_paths=1, 
                  max_path_length=3,
                  communities=None,
                  community_strengths=None):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.
        
        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
        ghetero.edata['eweight'] = eweight_dict

        # convert ghetero to ghomo and find paths
        ghomo = dgl.to_homogeneous(ghetero, edata=['eweight'])
        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]

        neg_path_score_func = get_neg_path_score_func(ghomo, 'eweight', [src_nid.item(), tgt_nid.item()],community=communities,community_strength=community_strengths)
        homo_paths = k_shortest_paths_with_max_length(ghomo, 
                                                       homo_src_nid, 
                                                       homo_tgt_nid,
                                                       weight=neg_path_score_func,
                                                       k=num_paths,
                                                       max_length=max_path_length,
                                                       community=communities,
                                                       community_strength=community_strengths)

        paths = []
        homo_nids_to_ntype_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)
    
        if len(homo_paths) > 0:
            for homo_path in homo_paths:
                hetero_path = []
                for i in range(1, len(homo_path)):
                    homo_u, homo_v = homo_path[i-1], homo_path[i]
                    hetero_u_ntype, hetero_u_nid = homo_nids_to_ntype_hetero_nids[homo_u] 
                    hetero_v_ntype, hetero_v_nid = homo_nids_to_ntype_hetero_nids[homo_v] 
                    can_etype = ntype_pairs_to_cannonical_etypes[(hetero_u_ntype, hetero_v_ntype)]    
                    hetero_path += [(can_etype, hetero_u_nid, hetero_v_nid)]
                paths += [hetero_path]

        else:
            # A rare case, no paths found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = ghetero.edges(etype=etype)  
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip([etype] * topk_edge_mask.sum().item(), u[topk_edge_mask].tolist(), v[topk_edge_mask].tolist()))                
            paths = [path]
        return paths


   
    def get_paths(self,
                  src_nid, 
                  tgt_nid, 
                  ghetero,
                  edge_mask_dict,
                  num_paths=1, 
                  max_path_length=3,
                  communities=None,
                  community_strengths=None):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.
        
        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
       
        
        
        
        ghetero.edata['eweight'] = eweight_dict

        # convert ghetero to ghomo and find paths
        ghomo = dgl.to_homogeneous(ghetero, edata=['eweight'])
        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]
        neg_path_score_func = get_neg_path_score_func(ghomo, 'eweight', [src_nid.item(), tgt_nid.item()],community=communities,community_strength=community_strengths)
        homo_paths = k_shortest_paths_with_max_length(ghomo, 
                                                       homo_src_nid, 
                                                       homo_tgt_nid,
                                                       weight=neg_path_score_func,
                                                       k=num_paths,
                                                       max_length=max_path_length,
                                                       community=communities,
                                                       community_strength=community_strengths)
        

        paths = []
        
        paths_com_num=[]
        
        paths_com=[]
        
        homo_nids_to_ntype_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)
    
    
        #communities=get_community(ghomo)
        #print(222,community_strengths)
        #print(333,communities)
        
    
        if len(homo_paths) > 0:
            for homo_path in homo_paths:
            
                com_num=len(set(communities[node] for node in homo_path)) 
                com=[communities[node] for node in homo_path]
                
                hetero_path = []
                for i in range(1, len(homo_path)):
                    homo_u, homo_v = homo_path[i-1], homo_path[i]
                    hetero_u_ntype, hetero_u_nid = homo_nids_to_ntype_hetero_nids[homo_u] 
                    hetero_v_ntype, hetero_v_nid = homo_nids_to_ntype_hetero_nids[homo_v] 
                    can_etype = ntype_pairs_to_cannonical_etypes[(hetero_u_ntype, hetero_v_ntype)]    
                    hetero_path += [(can_etype, hetero_u_nid, hetero_v_nid)]
                paths += [hetero_path]
                
                
                
                paths_com_num +=[com_num]
                paths_com +=[com]

        else:
            # A rare case, no paths found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = ghetero.edges(etype=etype) 
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip([etype] * topk_edge_mask.sum().item(), u[topk_edge_mask].tolist(), v[topk_edge_mask].tolist()))                
            
            
            paths = [path]
        return paths,paths_com_num,paths_com,homo_paths          

    
    def explain(self,  
                src_nid, 
                tgt_nid, 
                ghetero,
                membership,
                node_community_strengths,
                num_hops=2,
                prune_max_degree=-1,
                k_core=2, 
                num_paths=1, 
                max_path_length=3,
                prune_graph=True,
                with_path_loss=True,
                return_mask=False):
        
        """Return a path explanation of a predicted link
        
        Parameters
        ----------
        src_nid : int
            source node id

        tgt_nid : int
            target node id

        ghetero : dgl graph

        num_hops : int
            Number of hops to extract the computation graph, i.e. GNN # layers
            
        prune_max_degree : int
            If positive, prune the edges of graph nodes with degree larger than `prune_max_degree`
            If  -1, do nothing
            
        k_core : int 
            k for the the k-core graph extraction
            
        num_paths : int
            Number of paths for the postprocessing path extraction
            
        max_path_length : int
            Maximum length of paths for the postprocessing path extraction
        
        prune_graph : bool
            If true apply the max_degree and/or k-core pruning. For ablation. Default True.
            
        with_path_loss : bool
            If true include the path loss. For ablation. Default True.
            
        return_mask : bool
            If true return the edge mask in addition to the path. For AUC evaluation. Default False
        
        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)

        (optional) edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges
        """
        # Extract the computation graph (k-hop subgraph)
        (comp_g_src_nid, 
         comp_g_tgt_nid, 
         comp_g, 
         comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(self.src_ntype, 
                                                             src_nid, 
                                                             self.tgt_ntype, 
                                                             tgt_nid, 
                                                             ghetero, 
                                                             num_hops)
        # Learn the edge mask on the computation graph
        comp_g_edge_mask_dict,communities,community_strengths = self.get_edge_mask(comp_g_src_nid, 
                                                   comp_g_tgt_nid, 
                                                   comp_g, 
                                                   comp_g_feat_nids,
                                                   membership,
                                                   node_community_strengths,
                                                   prune_max_degree,
                                                   k_core,
                                                   prune_graph,
                                                   with_path_loss)

        # Extract paths 
        comp_g_paths = self.get_paths(comp_g_src_nid,
                                      comp_g_tgt_nid, 
                                      comp_g, 
                                      comp_g_edge_mask_dict, 
                                      num_paths, 
                                      max_path_length,
                                      communities,
                                      community_strengths)    
        
        
        # Covert the node id in computation graph to original graph
        paths = comp_g_paths_to_paths(comp_g, comp_g_paths)
        
        if return_mask:
            # return masks for easier evaluation
            return paths, comp_g_edge_mask_dict
        else:
            return paths 




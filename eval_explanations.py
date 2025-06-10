import torch
import numpy as np
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm

from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from utils import set_config_args, get_comp_g_edge_labels, get_comp_g_path_labels
from utils import hetero_src_tgt_khop_in_subgraph, eval_edge_mask_auc, eval_edge_mask_topk_path_hit,eval_edge_mask_auc_intra_inter,get_hetero_node_community_info

parser = argparse.ArgumentParser(description='Explain link predictor')
'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='aug_citation')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)
parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')

'''
Explanation args
'''
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops') 
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--eval_explainer_names', nargs='+', default=['pagelink'],
                    help='name of explainers to evaluate') 
parser.add_argument('--eval_path_hit', default=True, action='store_true', 
                    help='Whether to save the explanation') 
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')

args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')

if 'citation' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'

elif 'synthetic' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'item'    
elif 'ohgbn-imdbsyn' in args.dataset_name:
    args.src_ntype = 'director'
    args.tgt_ntype = 'actor'
elif 'rpddsyn' in args.dataset_name:
    args.src_ntype = 'a'
    args.tgt_ntype = 'f' 
elif 'HGBn-ACMsyn' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'subject'    
elif 'ohgbn-acmsyn' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'       
elif 'HGBn-IMDBsyn' in args.dataset_name:
    args.src_ntype = 'director'
    args.tgt_ntype = 'actor'
elif 'HGBn-DBLPsyn' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'term'        
elif 'ogbn-mag6syn' in args.dataset_name:
    args.src_ntype = 'author'
    args.tgt_ntype = 'institution'
    
    
                    
if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}

g, processed_g, pred_pair_to_edge_labels, pred_pair_to_path_labels = load_dataset(args.dataset_dir,
                                                                                  args.dataset_name,
                                                                                  args.valid_ratio,
                                                                                  args.test_ratio)
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g for g in processed_g]
encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'

state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)    

mp_membership,community_strengths, node_community_strengths,internal_pro,cross_pro  = get_hetero_node_community_info(mp_g)
print(111222333,[internal_pro,cross_pro ])

test_src_nids, test_tgt_nids = test_pos_g.edges()
comp_graphs = defaultdict(list)
comp_g_labels = defaultdict(list)
test_ids = range(test_src_nids.shape[0])
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

for i in tqdm(test_ids):
    # Get the k-hop subgraph
    src_nid, tgt_nid = test_src_nids[i], test_tgt_nids[i]
    comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids = hetero_src_tgt_khop_in_subgraph(args.src_ntype, 
                                                                                               src_nid,
                                                                                               args.tgt_ntype,
                                                                                               tgt_nid,
                                                                                               mp_g,
                                                                                               args.num_hops)

    with torch.no_grad():
        pred = model(comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids).sigmoid().item() > 0.5

    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        comp_graphs[src_tgt] = [comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids]

        # Get labels with subgraph nids and eids 
        edge_labels = pred_pair_to_edge_labels[src_tgt]
        comp_g_edge_labels = get_comp_g_edge_labels(comp_g, edge_labels)

        path_labels = pred_pair_to_path_labels[src_tgt]
        comp_g_path_labels = get_comp_g_path_labels(comp_g, path_labels)

        comp_g_labels[src_tgt] = [comp_g_edge_labels, comp_g_path_labels]

explanation_masks = {}
for explainer in args.eval_explainer_names:
    saved_explanation_mask = f'{explainer}_{args.saved_model_name}_pred_edge_to_comp_g_edge_mask'
    saved_file = Path.cwd().joinpath(args.saved_explanation_dir, saved_explanation_mask)
    with open(saved_file, "rb") as f:
        explanation_masks[explainer] = pickle.load(f)

print('Dataset:', args.dataset_name)
for explainer in args.eval_explainer_names:
    print(explainer)
    print('-'*30)
    pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]
    
    mask_auc_list = []
    mask_intro_auc_list = []
    mask_inter_auc_list = [] 
    mask_all_mean_pro_list=[]
    mask_intro_mean_pro_list=[]
    mask_inter_mean_pro_list=[]     
    sp_list=[]
    dp_list=[]
    eo_list=[]
    eo_pro_list=[]
    eo_tpr_list=[]
    eo_fpr_list=[] 
    eoo_list=[]    
    di_list=[]
    di_pro_list=[]
    ber_list=[]
    ber_pro_list=[]



       
    
    for src_tgt in comp_graphs:
        comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
        comp_g_edge_labels, comp_g_path_labels = comp_g_labels[src_tgt]
        comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
        #mask_auc = eval_edge_mask_auc(comp_g_edge_mask_dict, comp_g_edge_labels)
        mask_auc,mask_intro_auc,mask_inter_auc,y_score_mean,y_intro_score_mean,y_inter_score_mean,sp,dp,eo,eo_pro,EO_FPR,EO_TPR, EOO,DI,DI_probability,BER,BER_probability = eval_edge_mask_auc_intra_inter(comp_g_edge_mask_dict, comp_g_edge_labels,comp_g, comp_g_feat_nids,mp_membership)
        mask_auc_list += [mask_auc]
        mask_intro_auc_list += [mask_intro_auc]
        mask_inter_auc_list += [mask_inter_auc]
        mask_all_mean_pro_list += [y_score_mean]         
        mask_intro_mean_pro_list +=[y_intro_score_mean]  
        mask_inter_mean_pro_list +=[y_inter_score_mean]
                 
        sp_list+=[sp]
        dp_list+=[dp]
        eo_list+=[eo]
        eo_pro_list += [eo_pro]        
        eo_tpr_list+=[EO_TPR]
        eo_fpr_list +=[EO_FPR]
        eoo_list+=[EOO]                
        di_list += [DI]
        di_pro_list += [DI_probability]
        ber_list += [BER]
        ber_pro_list += [BER_probability]
        



      
    avg_auc = np.mean(mask_auc_list)
    if 0.5 in mask_intro_auc_list:
      mask_intro_auc_list.remove(0.5)
      avg_intro_auc = np.mean(mask_intro_auc_list)

    if 0.5 in mask_inter_auc_list:
      mask_inter_auc_list.remove(0.5)     
      avg_inter_auc = np.mean(mask_inter_auc_list)    


         
    avg_all_mean_pro = np.mean(mask_all_mean_pro_list)  
    avg_intro_mean_pro = np.mean(mask_intro_mean_pro_list)
    avg_inter_mean_pro = np.mean(mask_inter_mean_pro_list)  
        
    avg_sp = np.mean(sp_list)
    avg_dp = np.mean(dp_list)
    avg_eo = np.mean(eo_list)
    avg_eo_pro = np.mean(eo_pro_list)
    avg_eo_tpr = np.mean(eo_tpr_list)
    avg_eo_fpr = np.mean(eo_fpr_list)    
    avg_eoo = np.mean(eoo_list)
    avg_di = np.mean(di_list)
    avg_di_pro = np.mean(di_pro_list)
    avg_ber = np.mean(ber_list)
    avg_ber_pro = np.mean(ber_pro_list)    
    
    bias=abs(avg_inter_auc-avg_intro_auc)        
    bias_cr_pro=abs(avg_intro_mean_pro-avg_inter_mean_pro)    
    
    
    
        
    
    np.set_printoptions(precision=4, suppress=True)
    print(f'Average Mask-AUC: {avg_auc : .4f}')
    print(f'Average Intro-Mask-AUC: {avg_intro_auc : .4f}')
    print(f'Average Inter-Mask-AUC: {avg_inter_auc : .4f}')
    print(f'Average Bias: {bias : .4f}')
    print(f'Average Mask-all_mean_pro: {avg_all_mean_pro : .4f}')      
    print(f'Average Mask-intro_mean_pro: {avg_intro_mean_pro : .4f}')
    print(f'Average Mask-inter_mean_pro: {avg_inter_mean_pro : .4f}')
    print(f'Average Bias_cross_mean_pro: {bias_cr_pro : .4f}')              
    print(f'Average SP: {avg_sp : .4f}')
    print(f'Average DP: {avg_dp : .4f}')
    print(f'Average EO: {avg_eo : .4f}')
    print(f'Average EO_Pro: {avg_eo_pro : .4f}')        
    print(f'Average EO_TPR: {avg_eo_tpr : .4f}')  
    print(f'Average EO_FPR: {avg_eo_fpr : .4f}')
    print(f'Average EOO: {avg_eoo : .4f}')    
    print(f'Average DI: {avg_di : .4f}')
    print(f'Average DI_pro: {avg_di_pro : .4f}')
    print(f'Average BER: {avg_ber : .4f}')
    print(f'Average BER_pro: {avg_ber_pro : .4f}')
               


    
        
    print('-'*30, '\n')

if args.eval_path_hit:
    topks = [3, 5, 10, 20, 50, 100, 200]
    for explainer in args.eval_explainer_names:
        print(explainer)
        print('-'*30)
        pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

        explainer_to_topk_path_hit = defaultdict(list)
        for src_tgt in comp_graphs:
            comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
            comp_g_path_labels = comp_g_labels[src_tgt][1]
            comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
            topk_to_path_hit = eval_edge_mask_topk_path_hit(comp_g_edge_mask_dict, comp_g_path_labels, topks)

            for topk in topk_to_path_hit:
                explainer_to_topk_path_hit[topk] += [topk_to_path_hit[topk]]        

        # Take average
        explainer_to_topk_path_hit_rate = defaultdict(list)
        for topk in explainer_to_topk_path_hit:
            metric = np.array(explainer_to_topk_path_hit[topk])
            explainer_to_topk_path_hit_rate[topk] = metric.mean(0)

        # Print
        np.set_printoptions(precision=4, suppress=True)
        for k, hr in explainer_to_topk_path_hit_rate.items():
            print(f'k: {k :3} | Path HR: {hr.item(): .4f}')

        print('-'*30, '\n')

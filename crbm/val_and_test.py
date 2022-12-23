import pickle
import torch
import os

# from free_energy_cd_model_1_cond import CRBM

def metrics(v_orig_idx, v_p_idx):
    hit = len(set(v_orig_idx) & set(v_p_idx))
    if len(v_orig_idx) != 0:
        recall = hit / len(v_orig_idx)
    else:
        recall = 0
    if len(v_p_idx) != 0:
        precision = hit / len(v_p_idx)
    else:
        precision = 0
    if (recall != 0) & (precision != 0):
        f1_score = 2 * (precision*recall) / (precision+recall)
    else:
        f1_score = 0

    return recall, precision, f1_score

def val(val_main_batch, val_v_p, val_recall_list, val_precision_list, val_f1_score_list, CUDA, EPOCHS, epoch, save_path):
    for v_orig, v_p in zip(val_main_batch, val_v_p):
        # input query
        v_orig_idx = torch.nonzero(v_orig, as_tuple=True)[0].cpu().numpy()
        # top n prediction
        v_p_idx_top_1 = torch.argsort(v_p, dim=-1, descending=True)[:1].cpu().numpy()
        v_p_idx_top_5 = torch.argsort(v_p, dim=-1, descending=True)[:5].cpu().numpy()
        v_p_idx_top_10 = torch.argsort(v_p, dim=-1, descending=True)[:10].cpu().numpy()
        v_p_idx_top_20 = torch.argsort(v_p, dim=-1, descending=True)[:20].cpu().numpy()
        # recall, precision, f1_score at top n
        recall_top_1, precision_top_1, f1_score_top_1 = metrics(v_orig_idx, v_p_idx_top_1)
        recall_top_5, precision_top_5, f1_score_top_5 = metrics(v_orig_idx, v_p_idx_top_5)
        recall_top_10, precision_top_10, f1_score_top_10 = metrics(v_orig_idx, v_p_idx_top_10)
        recall_top_20, precision_top_20, f1_score_top_20 = metrics(v_orig_idx, v_p_idx_top_20)
        # export recommended queries and the hitted recommended queries.
        if epoch == EPOCHS - 1:
            f_name = save_path + 'prediction.txt'
            pred_f = open(f_name, "a+")
            pred_f.write('Ground truth:')
            pred_f.write(''.join(str(v_orig_idx)))
            pred_f.write('\n')
            pred_f.write('Top 10 recommendations:')
            pred_f.write(''.join(str(v_p_idx_top_10)))
            pred_f.write('\n')
            pred_f.write('recall: ')
            pred_f.write(str(recall_top_10))
            pred_f.write('\n')
            pred_f.write('precision: ')
            pred_f.write(str(precision_top_10))
            pred_f.write('\n')
            pred_f.write('f1 score: ')
            pred_f.write(str(f1_score_top_10))
            pred_f.write('\n\n')
            pred_f.close()
        # append recall list
        val_recall_list['top_1'].append(recall_top_1)
        val_recall_list['top_5'].append(recall_top_5)
        val_recall_list['top_10'].append(recall_top_10)
        val_recall_list['top_20'].append(recall_top_20)
        # append precision list
        val_precision_list['top_1'].append(precision_top_1)
        val_precision_list['top_5'].append(precision_top_5)
        val_precision_list['top_10'].append(precision_top_10)
        val_precision_list['top_20'].append(precision_top_20)
        # append f1_score list
        val_f1_score_list['top_1'].append(f1_score_top_1)
        val_f1_score_list['top_5'].append(f1_score_top_5)
        val_f1_score_list['top_10'].append(f1_score_top_10)
        val_f1_score_list['top_20'].append(f1_score_top_20)

    # pred.close()

    return val_recall_list, val_precision_list, val_f1_score_list

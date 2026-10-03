#!/usr/bin/env python
"""Focused numerical, sampler, real mixed-batch gradient, and serialization checks."""
import json,sys,tempfile,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'training/moment_detr_gmr'))
from models.moment_detr_trm_gmr_joint_v3.robust_auc import SemanticRobustAUCLoss
from models.moment_detr_trm_gmr_joint_v3.moment_detr_trm_gmr_joint import build_moment_detr_trm_gmr_joint
from models.moment_detr_trm_gmr_joint_v3.joint_loss import build_criterion_joint
from training.moment_detr_trm_gmr_joint_v3.config import BaseOptionsJoint
from training.moment_detr_trm_gmr_joint_v3.dataset import StartEndDatasetJoint,start_end_collate_joint,prepare_batch_inputs_joint
from training.moment_detr_trm_gmr_joint_v3.train import build_dataset_config_joint,set_seed,train_joint
from training.moment_detr_trm_gmr_joint_v3.semantic_groups import SemanticGroupBatchSampler,semantic_keys,group_auroc,eligible_validation_groups
from training.moment_detr_trm_gmr_joint_v3.evaluate import compute_mr_results_joint,eval_epoch_post_processing
import training.moment_detr_trm_gmr_joint_v3.infer


def main():
    set_seed(3407)
    loss=SemanticRobustAUCLoss(1,.1)
    scores=torch.tensor([2.,0.,-1.,1.],requires_grad=True)
    t={'exist_label':torch.tensor([1.,0.,1.,0.]),'group_action':torch.tensor([0,0,1,1]),'group_composition':torch.tensor([0,0,1,1]),'query_group':torch.tensor([0,0,1,1])}
    out=loss(scores,t)
    individual=torch.nn.functional.softplus(torch.tensor([-1.,3.]))
    expected=.1*torch.logsumexp(individual/.1,dim=0)
    assert torch.allclose(out['loss_robust_auc'],expected)
    shift=torch.tensor([100.,100.,-50.,-50.])
    assert torch.allclose(loss(scores+shift,t)['loss_robust_auc'],out['loss_robust_auc'],atol=1e-5)
    for y in [0.,1.]:
        tt=dict(t,exist_label=torch.full((4,),y));z=loss(scores,tt)
        assert z['loss_robust_auc'].item()==z['loss_same_semantic'].item()==0
        (z['loss_robust_auc']+z['loss_same_semantic']).backward(retain_graph=True)
    manager=BaseOptionsJoint();manager.parse();opt=manager.option
    opt.device='cuda';opt.drop_phrase=False;opt.exist_gate_thd=0.;opt.num_workers=0
    dataset=StartEndDatasetJoint(**build_dataset_config_joint(opt,opt.train_path,partition_filter=['S+','S-']))
    sampler=SemanticGroupBatchSampler(dataset.data)
    indices=next(iter(sampler));assert len(indices)==16
    assert next(iter(SemanticGroupBatchSampler(dataset.data)))==indices
    for epoch in [0,1]:
        sampler.set_epoch(epoch);batch_ids=next(iter(sampler));level=epoch%2
        groups=[]
        for k in range(0,16,4):
            block=[dataset.data[i] for i in batch_ids[k:k+4]]
            assert sum(r['exist_label'] for r in block)==2
            keys={semantic_keys(r)[level] for r in block};assert len(keys)==1;groups.extend(keys)
        assert len(set(groups))==4
    sampler.set_epoch(0);indices=next(iter(sampler))
    batch=start_end_collate_joint([dataset[i] for i in indices]);inputs,targets=prepare_batch_inputs_joint(batch[1],'cuda')
    for r,gt in zip(batch[0],targets['span_labels']):
        assert r['partition'] in ('S+','S-')
        if r['partition']=='S-':assert gt['spans'].numel()==0
    model=build_moment_detr_trm_gmr_joint(opt).cuda();criterion=build_criterion_joint(opt).cuda();model.train()
    assert not any('bg' in n or 'alpha_visual' in n for n,_ in model.named_parameters())
    output=model(**inputs)
    assert torch.equal(output['pred_exist_logits'],output['pred_exist_logits_semantic'])
    ranking=loss(output['pred_exist_logits'],targets);(ranking['loss_robust_auc']+ranking['loss_same_semantic']).backward()
    checked=[]
    for name,param in [('semantic_head',model.evidence_exist_head.mlp_exist_candidate[0].weight),('transformer',model.transformer.decoder.layers[0].linear1.weight),('phrase_matcher',model.phrase_matcher.phrase_proj.weight),('candidate_attention',model.candidate_attention.mlp_att[0].weight),('gate',model.gated_refinement.mlp_gate[0].weight)]:
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm()>0,name
        checked.append(name)
    model.zero_grad();outputs=model(**inputs);losses=criterion(outputs,targets)
    total=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
    torch.optim.AdamW(model.parameters(),lr=1e-4).step()
    negative=copy.copy(targets);negative['exist_label']=torch.zeros_like(targets['exist_label']);negative['span_labels']=[{'spans':g['spans'][:0]} for g in targets['span_labels']]
    neg_losses=criterion(model(**inputs),negative)
    for k in ['loss_span','loss_giou','loss_phrase_consistency','loss_phrase_exclusiveness','loss_robust_auc','loss_same_semantic']:assert neg_losses[k].item()==0,k
    model.eval();loader=DataLoader(Subset(dataset,indices),batch_size=16,collate_fn=start_end_collate_joint,num_workers=0)
    predictions,_=compute_mr_results_joint(-1,model,loader,opt)
    with tempfile.TemporaryDirectory() as tmp:
        opt.results_dir=tmp
        metrics,_=eval_epoch_post_processing(predictions,opt,batch[0],'tiny.jsonl')
        assert np.isfinite(metrics['brief']['MR-full-mAP'])
        loaded=[json.loads(l) for l in Path(tmp,'tiny.jsonl').read_text().splitlines()]
        assert all(a['pred_exist_logit']==b['pred_exist_logit'] and a['pred_exist_score']==b['pred_exist_score'] for a,b in zip(predictions,loaded))
    group=group_auroc(batch[0],predictions);assert group['worst_semantic_auroc'] is not None
    # One tiny Seen-only epoch exercises training statistics, validation, the
    # localization-constrained checkpoint path, and metadata serialization.
    class TinyDataset(Subset):
        def __init__(self, source, indices):
            super().__init__(source, indices)
            self.data = [source.data[i] for i in indices]
    tiny = TinyDataset(dataset, indices)
    tiny_sampler = SemanticGroupBatchSampler(tiny.data)
    for floor, expected_mode in [(0.0, 'constrained_worst_semantic_auroc'), (101.0, 'fallback_seen_mAP')]:
        with tempfile.TemporaryDirectory() as tmp:
            opt.results_dir = tmp; opt.ckpt_filepath = str(Path(tmp, 'best.ckpt'))
            opt.train_log_filepath = str(Path(tmp, 'train.log')); opt.eval_log_filepath = str(Path(tmp, 'val.log'))
            opt.semantic_manifest = str(Path(tmp, 'manifest.json'))
            # Direct helper test: even max_es_cnt=0 cannot truncate the epoch loop.
            # Formal CLI runs enforce n_epoch=50 and max_es_cnt=-1.
            opt.n_epoch = 3; opt.max_es_cnt = 0
            manifest = {'sampler_audit':tiny_sampler.audit(), 'localization_floor_mAP':floor,
                        'localization_reference_mAP':floor+1.0,
                        'validation_selection_groups':eligible_validation_groups(tiny.data,tiny.data,min_per_class=1)}
            Path(opt.semantic_manifest).write_text(json.dumps(manifest))
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,400)
            train_joint(model,criterion,optimizer,scheduler,tiny,tiny,opt)
            metadata = json.loads(Path(tmp,'training_meta.json').read_text())
            selected = json.loads(Path(tmp,'best_selection.json').read_text())
            map_selected = json.loads(Path(tmp,'best_mAP_selection.json').read_text())
            assert metadata['best_epoch']>=1 and metadata['training_status']=='completed'
            assert metadata['epochs_trained']==3 and metadata['early_stopping_enabled'] is False
            assert metadata['auc_valid_action_batch_fraction']==1.0
            assert metadata['selection_mode']==expected_mode==selected['selection_mode']
            assert selected['best_seen_val_mAP']==metadata['best_seen_val_mAP']
            assert map_selected['best_seen_val_mAP']==metadata['highest_seen_val_mAP']
            assert metadata['fallback_checkpoint_used']==(floor>100)
            for name in ['best.ckpt','best_mAP.ckpt']:
                assert Path(tmp,name).exists()
                model.load_state_dict(torch.load(Path(tmp,name),map_location='cuda',weights_only=False)['model'],strict=True)
    result={'status':'passed','device':'cuda','numeric_smooth_max':True,'semantic_offset_invariance':True,'all_positive_negative_graph_safe_zero':True,
      'balanced_sampler_deterministic':True,'real_mixed_batch_backward':True,'ranking_gradient_modules':checked,'semantic_only_existence':True,
      'empty_GT_loss_behavior':True,'optimizer_step':True,'real_evaluation_and_full_precision_serialization':True,'group_diagnostics':True,'tiny_train_validation_checkpoint_integration':True,'below_floor_checkpoint_saved':True,'forced_epoch_completion_no_early_stop':True,'nan_inf_detected':False}
    path=ROOT/'experiments/trm_gmr_joint_v3/SMOKE_TEST_RESULT.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

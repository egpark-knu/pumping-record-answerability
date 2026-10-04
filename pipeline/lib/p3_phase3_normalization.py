"""Exact installed decode/_preprocess call, intercepted before neural weights.
No modification of the installed model. Final context patch's rolled future input.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from timesfm3.torch import model as installed,util
SNAPSHOT=Path(__import__("os").environ.get("TIMESFM_SNAPSHOT", "timesfm-3.0-pytorch-43046b85"))
class Captured(Exception):
 def __init__(self,result):self.result=result

def capture_preprocessing(head,cov,H,device='mps'):
 head=np.asarray(head,np.float32);cov=np.asarray(cov,np.float32)
 if head.ndim!=2 or head.shape[1]!=1024 or cov.shape!=(len(head),2,1024+H) or H not in (10,30) or not np.isfinite(head).all() or not np.isfinite(cov).all():raise ValueError('Invalid observed head/forcing arrays')
 config=json.loads((SNAPSHOT/'config.json').read_text())
 shell=SimpleNamespace(**{k:config[k] for k in ('input_patch_len','output_patch_len','use_stitching','use_linear_detrending','linear_detrending_threshold','use_frozen_running_stats','value_clip')})
 shell.rolls=shell.output_patch_len//shell.input_patch_len;shell._stitching_extract_len=min(2*shell.input_patch_len,shell.output_patch_len)
 shell.pre_transformer_resblock=lambda x:x
 target=torch.from_numpy(head[:,None,:]).to(device);pf=torch.from_numpy(cov).to(device)
 original=torch.cat([target,pf[:,:,:1024]],dim=1)
 def boundary(inputs,freeze_after=None,patch_cpm_mask=None,return_aux_outputs=False):
  values=torch.clamp(torch.nan_to_num(inputs['values'],nan=0.),-shell.value_clip,shell.value_clip)
  masks=inputs['masks'].bool()
  res,_,_,(mu,std),n=installed.TimesFM3Torch._preprocess(shell,values,masks,inputs['patch_is_target'],freeze_after,patch_cpm_mask)
  last=1024//shell.input_patch_len-1
  # ResBlock input layout: current p + rolled future o, then masks p+o.
  future=res[:,:,last,shell.input_patch_len:shell.input_patch_len+H]
  context=values[:,:,:last+1,:].reshape(len(head),3,1024)
  out=dict(n=n[:,:,last],mean=mu[:,:,last],std=std[:,:,last],denominator=util._make_safe_for_division(std[:,:,last]),normalized_future=future,
           prepared_future=values[:,:,last+1:,:].reshape(len(head),3,-1)[:,:,:H],detrended=torch.any(context!=original,dim=-1),
           raw_context_std=torch.std(original,dim=-1,correction=0),last_context_patch=last)
  raise Captured({k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in out.items()})
 shell.forward=boundary
 try:
  installed.TimesFM3Torch.decode(shell,target,horizon=H,past_future_covariates=pf,mask=torch.zeros((len(head),1024),dtype=torch.bool,device=device))
 except Captured as capture:return capture.result
 raise RuntimeError('Capture boundary was not called')

#!/usr/bin/env python
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];split=sys.argv[1];out=root/'results/moment_detr_trm_gmr_joint_v3'/split
meta=json.loads((out/'training_meta.json').read_text())
if meta['best_epoch']>0:raise ValueError('Eligible checkpoint exists')
summary={'split':split,'status':'checkpoint_missing','reason':'No checkpoint was saved; this is a job failure, not a localization-floor exclusion','U_inference_run':False,**meta}
(out/'joint_v3_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))

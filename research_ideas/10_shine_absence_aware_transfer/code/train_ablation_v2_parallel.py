"""Single-arm launcher for unchanged frozen v2 training; isolate record filenames."""
import sys
from pathlib import Path
import train_ablation_v2 as train

if __name__=='__main__':
    index=sys.argv.index('--arms')
    arm=sys.argv[index+1]
    if arm not in ('E1_rotated_exist_only','SE1_full'):
        raise RuntimeError('Parallel launcher only handles queued E1/SE1')
    original_write=train.write
    def isolated_write(path,value):
        path=Path(path)
        if path.parent==train.ROOT/'records/ablation_v2' and (path.name.startswith('RESULTS_') or path.name.startswith('IMPORT_AUDIT_')):
            path=path.with_name(path.stem+'_'+arm+path.suffix)
        original_write(path,value)
    train.write=isolated_write
    train.main()

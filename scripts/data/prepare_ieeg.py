"""Adapt existing theta FIF + existing CLIP to the existing driven dataset API.

No NWB processing or feature extraction. The original iEEG decimator is reused.
FIF upstream whole-recording preprocessing is preserved, not train-only filtering.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from internal_dw.datasets.ieeg import _load_and_decimate, _chunk

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def align_features(times, feature_times, features):
    if np.any(np.diff(feature_times) <= 0) or feature_times[0] > times[0]:
        raise ValueError('Invalid stimulus timestamps or missing initial frame')
    return features[np.searchsorted(feature_times, times, side='right') - 1]


def normalize_splits(arrays, kind):
    train = arrays['train_' + kind].astype(np.float64)
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    scale = np.where(std == 0., 1., std)
    for split in ('train', 'validation', 'test'):
        key = split + '_' + kind
        arrays[key] = ((arrays[key].astype(np.float64) - mean) / scale).astype(np.float32)
        if not np.isfinite(arrays[key]).all():
            raise ValueError('Nonfinite normalized data')
    active = std.ravel() > 0
    result_std = arrays['train_' + kind].std(axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(result_std[active], 1., atol=1e-5)
    return dict(mean=mean.ravel().tolist(), std=std.ravel().tolist(),
                constant_dimensions=int((~active).sum()),
                normalized_std_median=float(np.median(result_std)))


def discover(root):
    groups={}
    for path in sorted(root.glob('P*CS_*_enc_macro_theta.fif')):
        match=re.fullmatch(r'(P(\d+)CS)_R\d+_enc_macro_theta\.fif',path.name)
        if not match: continue
        groups.setdefault(match.group(1),[]).append(path)
    if not groups: raise ValueError(f'No original theta FIF files under {root}')
    return groups


def load_clip(root,fps):
    features=np.load(root/'clip_projected.npy',allow_pickle=False)
    with (root/'clip_frames.csv').open(newline='') as stream:
        names=[r['frame'] for r in csv.DictReader(stream)]
    indices=np.array([int(re.fullmatch(r'frame_(\d+)\.png',n).group(1)) for n in names])
    if features.shape!=(len(indices),512) or not np.isfinite(features).all():
        raise ValueError('Invalid CLIP features')
    if len(indices)<2 or indices[0]!=0 or np.any(np.diff(indices)!=1):
        raise ValueError('Missing/nonconsecutive movie frames; inspect stimulus alignment')
    return features,indices/fps


def source_signature(paths,clip_root):
    files=list(paths)+[clip_root/'clip_projected.npy',clip_root/'clip_frames.csv']
    return [(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in files]


def adapt_subject(subject,paths,features,feature_times,fps):
    import mne
    runs=[]; channels=None
    for path in paths:
        raw=mne.io.read_raw_fif(path,preload=False,verbose='ERROR')
        current=list(raw.ch_names)
        if channels is not None and current!=channels:
            raise ValueError(f'Electrode order/set differs across repetitions: {subject}')
        channels=current
        if raw.n_times/raw.info['sfreq']>feature_times[-1]+2/fps:
            raise ValueError(f'FIF extends beyond movie coverage: {path}')
        raw.close()
        x,_=_load_and_decimate(path,step_ms=20.,max_channels=0)
        if not np.isfinite(x).all(): raise ValueError(f'Nonfinite FIF data: {path}')
        runs.append(x)
    # File data start at cropped movie onset. first_samp=5000 is prior padding,
    # NOT an additional 10-second stimulus offset.
    length=min(len(x) for x in runs)
    a,b=round(.7*length),round(.15*length)
    slices={'train':slice(0,a-256),'validation':slice(a,a+b-256),'test':slice(a+b,length)}
    arrays={}; provenance={}
    drive=align_features(np.arange(length)*.02,feature_times,features)
    for split,section in slices.items():
        arrays[split+'_state']=np.concatenate([_chunk(x[:length][section],1024) for x in runs])
        u=_chunk(drive[section],1024)
        arrays[split+'_drive']=np.concatenate([u for _ in runs])
        provenance[split]=dict(movie_start_seconds=section.start*.02,
                               movie_stop_seconds=section.stop*.02,chunks_per_run=len(u))
    stats={kind:normalize_splits(arrays,kind) for kind in ('state','drive')}
    meta=dict(dataset='ieeg_fif_visual_'+subject,source_subject=subject,channels=channels,
              runs=[str(p.resolve()) for p in paths],step_seconds=.02,
              split_policy='70/15/15 common movie-time boundaries within subject; 256-step pre-boundary gaps; no cross-run chunks',
              splits=provenance,upstream='existing FIF: recording-wide Morlet z-score and noncausal filtering preserved',
              normalization='per-subject train-only second-stage z-score, std==0 -> 1; already applied',
              normalization_parameters=stats,stimulus='existing 512-D CLIP projected; previous frame hold; visual only',
              stimulus_fps=fps)
    return arrays,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fif-root',type=Path,required=True)
    p.add_argument('--clip-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('probe_inputs/ieeg_cohort_v1'))
    p.add_argument('--subject',default=None,help='Optional one-subject preparation check, e.g. P41CS')
    p.add_argument('--fps',type=float,default=25.)
    args=p.parse_args()
    if args.fps<=0: p.error('fps must be positive')
    groups=discover(args.fif_root)
    if args.subject:
        args.subject = re.sub(r'^(?:sub-)?CS(\d+)$', r'P\1CS', args.subject)
        if args.subject not in groups: p.error('Unknown subject')
        groups={args.subject:groups[args.subject]}
    features,ft=load_clip(args.clip_root,args.fps)
    inventory=[]; reports=[]
    if args.subject:
        # Preparing one subject must not hide previously prepared participants.
        for name, rows in [('inventory.json', inventory), ('preparation_status.json', reports)]:
            path = args.out / name
            if path.is_file(): rows.extend(json.loads(path.read_text()))
    for subject,paths in groups.items():
        name='sub-CS'+subject[1:-2]
        inventory = [r for r in inventory if r['subject'] != name]
        reports = [r for r in reports if r['subject'] != name]
        inventory.append(dict(subject=name,source_subject=subject,runs=[str(p.resolve()) for p in paths]))
        target=args.out/'prepared'/(name+'.npz')
        sig=hashlib.sha256(json.dumps(dict(version=1,files=source_signature(paths,args.clip_root),fps=args.fps),sort_keys=True).encode()).hexdigest()
        report=dict(subject=name,status='failed')
        try:
            if target.exists():
                with np.load(target,allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata_json'].item()))
                    if meta.get('adapter_signature')!=sig: raise ValueError('Sources changed: choose a fresh --out')
                    shapes={k:list(z[k].shape) for k in z.files if k!='metadata_json'}
            else:
                arrays,meta=adapt_subject(subject,paths,features,ft,args.fps)
                meta['adapter_signature']=sig
                save_npz(target,**arrays,metadata_json=json.dumps(meta))
                shapes={k:list(v.shape) for k,v in arrays.items()}
            report.update(status='complete',shapes=shapes)
        except Exception as exc: report['error']=str(exc)
        reports.append(report)
        write_json(args.out/'preparation_status.json',reports)
        print(json.dumps(report),flush=True)
    write_json(args.out/'inventory.json',inventory)
    if any(r['status']!='complete' for r in reports): raise SystemExit(1)


if __name__=='__main__': main()

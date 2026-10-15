"""Sequence-local CMI cleaning. No fitted statistics or model features."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

PREFIXES = ('acc_', 'rot_', 'thm_', 'tof_')
CHANNELS = (['acc_'+a for a in 'xyz'] + ['rot_'+a for a in ('w','x','y','z')]
            + [f'thm_{i}' for i in range(1,6)]
            + [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)])


def clean_sequence(sequence):
    """Return (ordered raw rows, channel names, float32 values, bool observed).

    Whole-sequence offline transform; no labels, demographics or fitted statistics
    are used. False mask entries are replacements, not measured observations.
    """
    required = ['sequence_id','sequence_counter'] + CHANNELS
    if sequence.columns.duplicated().any():
        raise ValueError('Duplicate column names')
    absent = [c for c in required if c not in sequence]
    if absent:
        raise ValueError(f'Missing required columns: {absent}')
    if sequence.empty or sequence.sequence_id.isna().any():
        raise ValueError('Sequence must be nonempty with a nonmissing ID')
    if sequence.sequence_id.nunique() != 1:
        raise ValueError('Pass exactly one sequence')
    counter = pd.to_numeric(sequence.sequence_counter, errors='raise')
    if not np.isfinite(counter).all() or (counter < 0).any() or (counter % 1 != 0).any():
        raise ValueError('Sequence counters must be finite nonnegative integers')
    sequence = sequence.copy()
    sequence['sequence_counter'] = counter.astype('int64')
    data = sequence.sort_values('sequence_counter').reset_index(drop=True)
    if data.sequence_counter.duplicated().any():
        raise ValueError('Duplicate sequence counters')
    channels = CHANNELS.copy()
    values = data[channels].to_numpy(dtype=np.float64, copy=True)
    observed = np.isfinite(values)
    idx = {p: [i for i, c in enumerate(channels) if c.startswith(p)] for p in PREFIXES}
    if not observed[:, idx['acc_']].all():
        raise ValueError('Unexpected missing acceleration; requires explicit handling')
    # A quaternion is usable only if all components are finite and norm is positive.
    ri = [channels.index('rot_'+c) for c in ('w', 'x', 'y', 'z')]
    rotation = values[:, ri]
    with np.errstate(over='ignore', invalid='ignore'):
        norms = np.linalg.norm(rotation, axis=1)
    usable = np.isfinite(rotation).all(axis=1) & np.isfinite(norms) & (norms > 1e-8)
    observed[:, ri] = usable[:, None]
    rotation[usable] /= norms[usable, None]
    # Identity is a numerical placeholder, never a recovered orientation.
    rotation[~usable] = [1., 0., 0., 0.]
    values[:, ri] = rotation
    for i in idx['thm_']:
        x = pd.Series(np.where(observed[:, i], values[:, i], np.nan), index=data.sequence_counter.to_numpy())
        # Offline inference receives the whole sequence. Never cross sequence boundaries.
        values[:, i] = x.interpolate(method='index', limit_direction='both').fillna(0).to_numpy()
    ti = idx['tof_']
    observed[:, ti] &= values[:, ti] != -1
    values[:, ti] = np.where(observed[:, ti], values[:, ti], 0)
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite cleaned values')
    return data, channels, values, observed


def checks():
    base = pd.DataFrame({'sequence_id':['X']*3, 'sequence_counter':[2,0,1],
        'acc_x':[1.,2.,3.], 'acc_y':[0.]*3, 'acc_z':[0.]*3,
        'rot_w':[np.nan]*3, 'rot_x':[np.nan]*3, 'rot_y':[np.nan]*3, 'rot_z':[np.nan]*3,
        'thm_1':[30.,20.,np.nan], 'thm_2':[np.nan]*3, 'tof_1_v0':[-1.,0.,np.nan]})
    base = base.reindex(columns=['sequence_id','sequence_counter']+CHANNELS)
    ordered, names, x, mask = clean_sequence(base)
    assert ordered.sequence_counter.tolist()==[0,1,2]
    assert np.allclose(x[:, names.index('thm_1')], [20,25,30])
    assert not mask[:, names.index('thm_2')].any()
    assert mask[:, names.index('tof_1_v0')].tolist()==[True,False,False]
    assert np.allclose(x[:, names.index('rot_w')], 1)
    assert not mask[:, names.index('rot_w')].any()
    assert np.allclose(x[:, names.index('acc_x')], [2,3,1])
    base.loc[:, 'rot_w']=2.; base.loc[:, ['rot_x','rot_y','rot_z']]=0.
    _, names, x, mask=clean_sequence(base)
    assert np.allclose(x[:, names.index('rot_w')],1) and mask[:, names.index('rot_w')].all()
    print('Synthetic checks passed: sorting, masks, interpolation, zero distances, rotation fallback', flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--output', default='processed_data')
    parser.add_argument('--chunk-size', type=int, default=20000,
                        help='CSV rows read at once (positive integer)')
    args=parser.parse_args()
    if args.chunk_size < 1: parser.error('--chunk-size must be positive')
    checks()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new output directory')
    with zipfile.ZipFile(args.zip) as z:
        absent = {'train.csv','train_demographics.csv'} - set(z.namelist())
        if absent: raise ValueError(f'Archive lacks required files: {sorted(absent)}')
    out.mkdir(parents=True, exist_ok=True)
    pending=[]; metas=[]; ids=[]; index=[]; examples=[]; total=0; part=0; names=[]
    groups = {p: {'unobserved_cells':0, 'sequences_with_unobserved':0} for p in PREFIXES}
    processed_ids=set(); source_rows=0

    def flush():
        nonlocal part
        if not pending: return
        stem=f'part_{part:03d}'
        np.savez_compressed(out/(stem+'.npz'), values=np.concatenate([p[0] for p in pending]), observed=np.concatenate([p[1] for p in pending]), channels=np.array(names))
        pd.concat(metas, ignore_index=True).to_csv(out/(stem+'_metadata.csv'),index=False)
        offset=0
        for sid, (x, _) in zip(ids,pending):
            index.append({'sequence_id':sid,'part':stem+'.npz','start_row':offset,'rows':len(x)})
            offset+=len(x)
        pending.clear(); metas.clear(); ids.clear(); part+=1

    def consume(group):
        nonlocal total, names
        data,names,x,mask=clean_sequence(group)
        sid=data.sequence_id.iloc[0]
        if sid in processed_ids:
            raise ValueError(f'Noncontiguous sequence {sid}; sort source CSV by sequence first')
        processed_ids.add(sid)
        for c in ('subject','gesture'):
            if data[c].isna().any() or data[c].nunique()!=1:
                raise ValueError(f'{sid}: missing or inconsistent {c}')
        if data.subject.iloc[0] not in demographic_subjects:
            raise ValueError(f'{sid}: subject missing from demographics')
        summary={'sequence_id':sid,'subject':data.subject.iloc[0],'gesture':data.gesture.iloc[0],'rows':len(data)}
        for p in PREFIXES:
            cols=[i for i,c in enumerate(names) if c.startswith(p)]
            count=int((~mask[:,cols]).sum())
            groups[p]['unobserved_cells']+=count
            groups[p]['sequences_with_unobserved']+=bool(count)
            summary[p+'unobserved_cells']=count
        wanted = ('complete' if not any(summary[p+'unobserved_cells'] for p in PREFIXES) else
                  'missing_rotation' if summary['rot_unobserved_cells'] else 'missing_other_sensor')
        if wanted not in [e['example_type'] for e in examples]:
            summary['example_type']=wanted; examples.append(summary)
            preview=data[['sequence_id','sequence_counter','subject','gesture']].copy()
            for c in ['acc_x','rot_w','thm_1','tof_1_v0']:
                i=names.index(c); preview[c+'_raw']=data[c]; preview[c+'_clean']=x[:,i]; preview[c+'_observed']=mask[:,i]
            preview.to_csv(out/('example_'+wanted+'.csv'),index=False)
        pending.append((x,mask)); metas.append(data[[c for c in data if c not in names]])
        ids.append(sid); total+=len(data)
        if len(pending)>=100:
            flush(); print(f'Cleaned {total:,} rows; saved {part} parts',flush=True)

    with zipfile.ZipFile(args.zip) as z:
        demographics=pd.read_csv(z.open('train_demographics.csv'))
        if demographics.subject.isna().any() or demographics.subject.duplicated().any():
            raise ValueError('Demographic subject IDs must be present and unique')
        demographic_subjects=set(demographics.subject)
        demographics.to_csv(out/'train_demographics.csv',index=False)
        tail=None
        for chunk in pd.read_csv(z.open('train.csv'),chunksize=args.chunk_size):
            source_rows+=len(chunk)
            if chunk.empty: continue
            if chunk.sequence_id.isna().any(): raise ValueError('Missing sequence ID')
            if tail is not None: chunk=pd.concat([tail,chunk],ignore_index=True)
            # Split contiguous runs, not IDs: A,B,A must fail rather than merge.
            identifiers=chunk.sequence_id.to_numpy()
            starts=np.r_[0, np.flatnonzero(identifiers[1:]!=identifiers[:-1])+1]
            for begin,end in zip(starts[:-1],starts[1:]):
                consume(chunk.iloc[begin:end])
            tail=chunk.iloc[starts[-1]:].copy()
        if tail is not None: consume(tail)
    flush()
    if source_rows==0: raise ValueError('Training CSV contains no data rows')
    if total!=source_rows: raise ValueError('Output row count does not match input')
    pd.DataFrame(index).to_csv(out/'sequence_index.csv',index=False)
    result={'status':'complete','source_zip':str(Path(args.zip).resolve()),'rows':total,'sequences':len(index),'parts':part,'channels':len(names),'storage_dtype':'float32','groups':groups,'examples':examples}
    (out/'cleaning_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()

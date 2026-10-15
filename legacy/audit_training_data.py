"""Read-only, chunked audit of the original CMI training CSV in its ZIP."""
import argparse
import collections
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    args = parser.parse_args()
    out = Path('audit_results')
    out.mkdir(exist_ok=True)
    missing = collections.Counter()
    invalid = collections.Counter()
    seqs = {}
    seen_hashes = set()
    seen_ids = set()
    duplicates = duplicate_ids = total = 0
    with zipfile.ZipFile(args.zip) as archive:
        demographics = pd.read_csv(archive.open('train_demographics.csv'))
        for chunk in pd.read_csv(archive.open('train.csv'), chunksize=20000):
            columns = list(chunk.columns)
            sensor = [c for c in columns if c.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
            tof = [c for c in sensor if c.startswith('tof_')]
            missing.update(chunk.isna().sum().to_dict())
            invalid.update(chunk[tof].eq(-1).sum().to_dict())
            for h in pd.util.hash_pandas_object(chunk, index=False):
                duplicates += h in seen_hashes
                seen_hashes.add(h)
            for rid in chunk.row_id:
                duplicate_ids += rid in seen_ids
                seen_ids.add(rid)
            for sid, group in chunk.groupby('sequence_id', sort=False):
                s = seqs.setdefault(sid, {'rows': 0, 'subjects': set(), 'gestures': set(), 'counters': [], 'missing': collections.Counter(), 'minus_one': 0})
                s['rows'] += len(group)
                s['subjects'].update(group.subject.dropna())
                s['gestures'].update(group.gesture.dropna())
                s['counters'].extend(group.sequence_counter.tolist())
                s['missing'].update({p: int(group[[c for c in sensor if c.startswith(p)]].isna().sum().sum()) for p in ('acc_', 'rot_', 'thm_', 'tof_')})
                s['minus_one'] += int(group[tof].eq(-1).sum().sum())
            total += len(chunk)
            print(f'Audited {total:,} rows', flush=True)
    records = []
    issues = []
    for sid, s in seqs.items():
        counters = s['counters']
        unique = set(counters)
        gaps = max(unique) - min(unique) + 1 - len(unique)
        duplicate_counters = len(counters) - len(unique)
        rec = {'sequence_id': sid, 'rows': s['rows'], 'subject': '|'.join(sorted(s['subjects'])), 'gesture': '|'.join(sorted(s['gestures'])), 'counter_min': min(unique), 'counter_max': max(unique), 'counter_gaps': gaps, 'duplicate_counters': duplicate_counters, 'out_of_order': any(b < a for a, b in zip(counters, counters[1:])), 'tof_minus_one_cells': s['minus_one']}
        for p, n in s['missing'].items():
            rec[p+'missing_cells'] = n
        records.append(rec)
        if len(s['subjects']) != 1 or len(s['gestures']) != 1 or gaps or duplicate_counters or rec['out_of_order']:
            issues.append(rec)
    sequences = pd.DataFrame(records)
    sequences.to_csv(out/'sequence_audit.csv', index=False)
    pd.DataFrame({'column': columns, 'missing_cells': [missing[c] for c in columns], 'missing_percent': [100*missing[c]/total for c in columns], 'tof_minus_one_cells': [invalid[c] for c in columns]}).to_csv(out/'column_audit.csv', index=False)
    counts = sequences.gesture.value_counts().rename_axis('gesture').reset_index(name='sequences')
    counts.to_csv(out/'gesture_counts.csv', index=False)
    summary = {'source_zip': str(Path(args.zip).resolve()), 'rows': total, 'columns': len(columns), 'sequences': len(sequences), 'subjects': int(sequences.subject.nunique()), 'gestures': int(sequences.gesture.nunique()), 'exact_duplicate_rows_hash_check': duplicates, 'duplicate_row_ids': duplicate_ids, 'sequences_with_duplicate_counters': int((sequences.duplicate_counters>0).sum()), 'sequences_with_counter_gaps': int((sequences.counter_gaps>0).sum()), 'out_of_order_sequences': int(sequences.out_of_order.sum()), 'inconsistent_subject_sequences': sum(len(s['subjects'])!=1 for s in seqs.values()), 'inconsistent_gesture_sequences': sum(len(s['gestures'])!=1 for s in seqs.values()), 'sequence_length': {str(q): float(sequences.rows.quantile(q)) for q in (0,.1,.5,.9,.95,.99,1)}, 'demographic_rows': len(demographics), 'duplicate_demographic_subjects': int(demographics.subject.duplicated().sum()), 'subjects_without_demographics': sorted(set(sequences.subject)-set(demographics.subject)), 'sensor_groups': {}}
    for prefix in ('acc_', 'rot_', 'thm_', 'tof_'):
        cols = [c for c in sensor if c.startswith(prefix)]
        m = sequences[prefix+'missing_cells']
        summary['sensor_groups'][prefix] = {'columns': len(cols), 'missing_cells': int(m.sum()), 'missing_percent': float(100*m.sum()/(total*len(cols))), 'sequences_with_missing': int((m>0).sum()), 'fully_missing_sequences': int((m==sequences.rows*len(cols)).sum())}
    summary['tof_minus_one_cells'] = int(sum(invalid.values()))
    summary['tof_minus_one_percent_of_all_tof_cells'] = 100*sum(invalid.values())/(total*len(tof))
    summary['both_thm_tof_fully_missing_sequences'] = int(((sequences.thm_missing_cells==sequences.rows*5)&(sequences.tof_missing_cells==sequences.rows*320)).sum())
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'sequence_issues.json').write_text(json.dumps(issues, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

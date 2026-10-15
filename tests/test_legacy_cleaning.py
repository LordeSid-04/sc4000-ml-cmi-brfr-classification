import unittest
import contextlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy'))
from preprocess_cmi import CHANNELS, clean_sequence, main


def recording():
    frame = pd.DataFrame(0., index=range(3), columns=CHANNELS)
    frame['rot_w'] = 1.
    frame['sequence_id'] = 'EXAMPLE'
    frame['sequence_counter'] = [0, 1, 2]
    return frame


class CleaningTests(unittest.TestCase):
    def test_temperature_and_masks(self):
        frame=recording(); frame['thm_1']=[20,np.nan,30]; frame['thm_2']=np.nan
        _,c,x,m=clean_sequence(frame)
        np.testing.assert_allclose(x[:,c.index('thm_1')],[20,25,30])
        self.assertEqual(m[:,c.index('thm_1')].tolist(),[True,False,True])
        self.assertFalse(m[:,c.index('thm_2')].any())
        self.assertTrue((x[:,c.index('thm_2')]==0).all())

    def test_distance_zero_and_invalid(self):
        frame=recording(); frame['tof_1_v0']=[0,-1,np.nan]
        _,c,x,m=clean_sequence(frame)
        self.assertEqual(m[:,c.index('tof_1_v0')].tolist(),[True,False,False])
        self.assertTrue((x[:,c.index('tof_1_v0')]==0).all())

    def test_rotation(self):
        frame=recording(); frame['rot_w']=[2,np.nan,0]
        _,c,x,m=clean_sequence(frame)
        np.testing.assert_allclose(x[:,c.index('rot_w')],1)
        self.assertEqual(m[:,c.index('rot_w')].tolist(),[True,False,False])

    def test_input_unchanged_and_order(self):
        frame=recording().iloc[[2,0,1]]; original=frame.copy(deep=True)
        _,c,x,m=clean_sequence(frame)
        pd.testing.assert_frame_equal(frame,original)
        ordered,c2,x2,m2=clean_sequence(frame[frame.columns[::-1]])
        self.assertEqual(ordered.sequence_counter.tolist(),[0,1,2])
        self.assertEqual(c,c2); np.testing.assert_array_equal(x,x2); np.testing.assert_array_equal(m,m2)

    def test_label_independence(self):
        frame=recording(); _,_,x,m=clean_sequence(frame)
        frame['gesture']='Any label'; frame['subject']='Some person'
        _,_,x2,m2=clean_sequence(frame)
        np.testing.assert_array_equal(x,x2); np.testing.assert_array_equal(m,m2)

    def test_reject_invalid_inputs(self):
        cases=[]
        f=recording(); f['sequence_counter']=[0,0,1]; cases.append(f)
        f=recording(); f.loc[0,'sequence_id']='Other'; cases.append(f)
        f=recording(); f.loc[0,'acc_x']=np.nan; cases.append(f)
        f=recording(); f.loc[0,'sequence_counter']=-1; cases.append(f)
        f=recording().drop(columns='tof_5_v63'); cases.append(f)
        f=recording(); f.loc[0,'sequence_id']=None; cases.append(f)
        cases.append(recording().iloc[:0])
        for f in cases:
            with self.subTest(columns=len(f.columns),rows=len(f)):
                with self.assertRaises(ValueError): clean_sequence(f)

    def test_temperature_counter_spacing_and_edges(self):
        f=recording(); f['sequence_counter']=[0,1,4]; f['thm_1']=[20,np.nan,40]
        _,c,x,m=clean_sequence(f)
        np.testing.assert_allclose(x[:,c.index('thm_1')],[20,25,40])
        f['thm_1']=[np.nan,25,np.nan]
        _,c,x,m=clean_sequence(f)
        np.testing.assert_allclose(x[:,c.index('thm_1')],25)

    def test_extreme_rotation_and_absent_sensor_groups(self):
        f=recording(); f['rot_w']=1e308
        for c in CHANNELS:
            if c.startswith(('tof_','thm_')):f[c]=np.nan
        _,c,x,m=clean_sequence(f)
        self.assertTrue(np.isfinite(x).all())
        self.assertFalse(m[:,[i for i,n in enumerate(c) if n.startswith(('tof_','thm_','rot_'))]].any())

    def run_cli(self,frame,root,output='out',chunk_size=2):
        archive=root/'input.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('train.csv',frame.to_csv(index=False))
            z.writestr('train_demographics.csv','subject,age\nP,20\n')
        with patch('sys.argv',['preprocess_cmi.py','--zip',str(archive),'--output',str(root/output),'--chunk-size',str(chunk_size)]),contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_cli_chunk_boundaries_and_readme_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=recording(); f['subject']='P'; f['gesture']='G'
            second=f.copy();second['sequence_id']='SECOND'
            frame=pd.concat([f,second],ignore_index=True)
            self.run_cli(frame,root)
            summary=json.loads((root/'out/cleaning_summary.json').read_text())
            self.assertEqual((summary['rows'],summary['sequences'],summary['status']),(6,2,'complete'))
            index=pd.read_csv(root/'out/sequence_index.csv')
            for r in index.itertuples():
                with np.load(root/'out'/r.part,allow_pickle=False) as part:
                    selected=part['values'][r.start_row:r.start_row+r.rows]
                    self.assertEqual(selected.shape,(3,332))
            with self.assertRaises(ValueError):self.run_cli(frame,root)

    def test_cli_reject_noncontiguous_sequences_and_unknown_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=recording();f['subject']='P';f['gesture']='G'
            f['sequence_id']=['A','B','A']
            with self.assertRaisesRegex(ValueError,'Noncontiguous'):self.run_cli(f,root,chunk_size=100)
            self.assertFalse((root/'out/cleaning_summary.json').exists())
            f['sequence_id']='A';f['subject']='UNKNOWN'
            with self.assertRaisesRegex(ValueError,'demographics'):self.run_cli(f,root,output='unknown')

    def test_cli_part_boundary_and_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=recording();f['subject']='P';f['gesture']='G'
            frames=[]
            for i in range(101):
                g=f.copy();g['sequence_id']=f'S{i:03d}';frames.append(g)
            self.run_cli(pd.concat(frames,ignore_index=True),root,chunk_size=17)
            index=pd.read_csv(root/'out/sequence_index.csv')
            self.assertEqual(len(index),101)
            self.assertEqual(index.part.nunique(),2)
            self.assertEqual(index.iloc[-1].start_row,0)
            self.assertEqual(index.iloc[-1].rows,3)
            with self.assertRaisesRegex(ValueError,'no data rows'):
                self.run_cli(f.iloc[:0],root,output='empty')
            self.assertFalse((root/'empty/cleaning_summary.json').exists())


if __name__=='__main__': unittest.main()

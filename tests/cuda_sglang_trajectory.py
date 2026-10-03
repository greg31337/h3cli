#!/usr/bin/env python3
"""Require every free-running AV velocity and sampler update, including C2/50."""
import argparse
import hashlib
import json
from pathlib import Path
from cuda_sglang_compare import metric,native,oracle,pack_video,pack_audio


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('oracle',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();specs=[json.loads((d/'command.json').read_text()) for d in (a.oracle,a.native)]
    for key in ('case','frames','evaluations'):
        if specs[0][key]!=specs[1][key]:raise ValueError('incompatible '+key)
    if any('H3_TEST_NATIVE_TEACHER_DIR=' in v for v in specs[1]['overrides']):raise ValueError('teacher-forced run is not free-running')
    for d in (a.oracle,a.native):
        if json.loads((d/'result.json').read_text())['status']!='media_valid':raise ValueError('incomplete render')
    contract_path=Path(__file__).with_name('cuda_sglang_contract.json');contract=json.loads(contract_path.read_text())
    results=[]
    def target_state(step,modality):
        value=oracle(a.oracle/'capture',f'step-{step:03d}.{modality}')
        prefix=json.loads((a.oracle/'capture'/f'positive.{modality}_target_start.json').read_text())
        if not isinstance(prefix,int) or prefix<0 or prefix>=value.shape[0]:raise ValueError('invalid target state range')
        # Conditioning rows remain in the oracle sampler state. Native exports
        # the updateable target only; velocities are already target-only.
        return value[prefix:]
    def target_velocity(step,index,modality):
        value=oracle(a.oracle/'capture',f'step-{step:03d}.velocity.{index}')
        # The pinned model filters video outputs before its final head but
        # retains reference audio rows; the denoise loop slices those away.
        if modality=='audio':
            prefix=json.loads((a.oracle/'capture'/'positive.audio_target_start.json').read_text())
            if not isinstance(prefix,int) or prefix<0 or prefix>=value.shape[0]:
                raise ValueError('invalid target velocity range')
            value=value[prefix:]
        return value
    for step in range(specs[0]['evaluations']):
        for index,(modality,pack) in enumerate((('video',pack_video),('audio',pack_audio))):
            for kind,tail,source in (('full_velocity','velocity',f'velocity.{index}'),('trajectory','latent',modality)):
                label=f'step-{step}.{modality}-'+('state' if tail=='latent' else tail)
                try:
                    row=dict(name=label,kind=kind,**metric(pack(native(a.native/'steps',f'step-{step+1:03d}-{modality}-{tail}.f32')),
                        target_state(step,modality) if tail=='latent' else target_velocity(step,index,modality)))
                except (ValueError,OSError,KeyError) as e:row=dict(name=label,kind=kind,error=str(e))
                g=contract['gates'][kind]
                row['passed']=bool(row.get('finite') and row.get('compatible') and row['relative_l2']<=g['relative_l2_max'] and row['cosine']>=g['cosine_min'])
                results.append(row);print(json.dumps(row),flush=True)
    imported=any(v.startswith(('H3_TEST_SGLANG_INPUT_DIR=','H3_TEST_SGLANG_CONDITION_DIR=')) and v.split('=',1)[1] for v in specs[1]['overrides'])
    report=dict(comparison='imported_conditioning_trajectory' if imported else 'free_running_trajectory',
        native_conditioning_qualified=not imported,
        case=specs[0]['case'],evaluations=specs[0]['evaluations'],
        oracle=str(a.oracle),native=str(a.native),results=results,passed=all(r['passed'] for r in results),
        first_failing_boundary=next((r['name'] for r in results if not r['passed']),None),
        bitwise_equal=all(r.get('exact',False) for r in results),contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()

from pathlib import Path
import os,json,time,subprocess
os.chdir('/data/hlou/pigment-pour-study')
os.environ.update(PYTHONPATH='/data/hlou/pigment-pour-study/wall-x',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
run=Path('runs/cylinder-hold-correction-bc-v1');parent=Path('runs/cylinder-odometry-bc-v1/checkpoints/0_2040')
log=Path('results/cylinder-hold-correction-bc-v1.log');driver=json.loads((run/'execution-process.json').read_text())['pid']
py='.venv-wall/bin/python';sim='pigment-pour-sim/simulation/';history=[]
status=run/'evaluation-status.json'
def save():status.write_text(json.dumps(history,indent=2)+'\n')
def score(report):
    c=report['episodes'][0];g=c['final_task_gates']
    stage=sum(bool(g[k]) for k in ['held','lifted','retreated','close_to_head'])
    return (int(bool(report['successes'])),stage,min(.20,max(0.,g.get('base_retreat_m',0.))) if g['lifted'] else 0.,min(.15,c['max_lifted_m']),-c['max_slip_m'])
def call(argv):
    print('START',argv,flush=True);p=subprocess.Popen(argv)
    row=dict(argv=argv,pid=p.pid,started=time.time());history.append(row);save()
    row.update(exit_code=p.wait(),finished=time.time());save()
    if row['exit_code']:raise SystemExit(row['exit_code'])
best_model=parent;best_summary=Path('runs/cylinder-odometry-step2040-single-v2/summary.json');best_score=score(json.loads(best_summary.read_text()))
(run/'checkpoint-retention-policy.json').write_text(json.dumps(dict(parent=str(parent),keep='Parent, latest complete, best provisional development-case weights; only latest optimizer',selection_scope='One development case, not a multi-case model ranking',score_order=['full single success','held/lifted/retreated/close gates','retreat capped20cm','lift capped15cm','lower slip'],all_other_checkpoints_require_completed_evaluation_before_retirement=True),indent=2)+'\n')
visited=[]
for step in range(128,1025,128):
    marker=f'at global_step={step}'
    while not log.exists() or marker not in log.read_text(errors='replace'):
        try:cmd=Path(f'/proc/{driver}/cmdline').read_bytes()
        except FileNotFoundError:cmd=b''
        if b'cylinder-hold-correction-bc-v1-job.py' not in cmd:
            raise RuntimeError(f'Training driver terminal before expected checkpoint{step}; inspect status rather than restart')
        time.sleep(15)
    model=run/f'checkpoints/0_{step}';out=Path(f'runs/cylinder-hold-correction-step{step}-single-v1')
    call([py,sim+'audit_wall_checkpoint.py','--base',str(parent),'--checkpoint',str(model),'--output',str(run/f'checkpoint-{step}-weight-audit.json')])
    call([py,sim+'run_wall_cylinder.py','--model',str(model),'--config',str(run/'train.yml'),'--norm-key','cylinder_official_v2','--controller-config',sim+'configs/cylinder_official_teacher_dev_v1.json','--volume','25','--seed','100','--output',str(out),'--execute-steps','8','--max-seconds','90','--record'])
    summary=out/'summary.json';r=json.loads(summary.read_text())
    if not r['evaluation_complete']:raise RuntimeError('Incomplete policy evaluation')
    visited.append((step,model))
    if score(r)>best_score:best_score=score(r);best_model=model;best_summary=summary
    (run/'best-development-case.json').write_text(json.dumps(dict(model=str(best_model),summary=str(best_summary),score=best_score,scope='Provisional single development case, all full-suite gates still required'),indent=2)+'\n')
    active=subprocess.run(['ps','-eo','args='],capture_output=True,text=True).stdout
    retired=[]
    for previous,checkpoint in visited:
        if previous==step:continue
        if any('python' in line and str(checkpoint) in line for line in active.splitlines()):continue
        for name in ['model.safetensors','optimizer.pt']:
            if name=='model.safetensors' and checkpoint==best_model:continue
            file=checkpoint/name
            if file.exists() and not file.is_symlink():
                row=dict(path=str(file),bytes=file.stat().st_size,reason='Superseded complete/evaluated checkpoint, not selected best; latest optimizer retained',time=time.time());file.unlink();retired.append(row)
    if retired:
        with (run/'checkpoint-retention.jsonl').open('a') as f:
            for row in retired:f.write(json.dumps(row)+'\n')
        print('PRUNED_OLD_ARTIFACTS',len(retired),flush=True)
    if r['successes']:
        (run/'candidate-ready.json').write_text(json.dumps(dict(step=step,model=str(model),single_summary=str(summary),requires=['original8','development12','untouched_independent12'],goal_complete=False),indent=2)+'\n')
        print('FULL_SINGLE_CANDIDATE_READY',step,flush=True);break
print('HOLD_CORRECTION_EVALUATION_QUEUE_FINISHED',flush=True)

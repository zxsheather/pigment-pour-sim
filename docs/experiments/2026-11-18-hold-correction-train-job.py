from pathlib import Path
import subprocess,os,json,time
os.chdir('/data/hlou/pigment-pour-study')
os.environ.update(PYTHONPATH='/data/hlou/pigment-pour-study/wall-x',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
run=Path('runs/cylinder-hold-correction-bc-v1');status=Path('results/cylinder-hold-correction-bc-v1-status.json');history=[]
def call(argv):
    print('START',argv,flush=True);p=subprocess.Popen(argv)
    r=dict(argv=argv,pid=p.pid,started=time.time());history.append(r);status.write_text(json.dumps(history,indent=2)+'\n')
    r.update(exit_code=p.wait(),finished=time.time());status.write_text(json.dumps(history,indent=2)+'\n')
    if r['exit_code']:raise SystemExit(r['exit_code'])
call(['.venv-wall/bin/python','-m','torch.distributed.run','--nproc_per_node=1','--master_port=29924','pigment-pour-sim/simulation/validate_wall_data.py','--config',str(run/'train.yml'),'--check-episode','51','--output',str(run/'data-validation.json')])
call(['.venv-wall/bin/python','-m','torch.distributed.run','--nproc_per_node=1','--master_port=29926','wall-x/wall_x/trainer/fsdp_trainer/train_fsdp.py','--config',str(run/'train.yml')])
print('TARGETED_HOLD_CORRECTION_TRAINING_FINISHED',flush=True)

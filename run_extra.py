import os, sys, subprocess, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
MODEL=os.environ.get('MODEL','audioseal')
BASE=os.path.expanduser('~/wm_compare'); AUDIO=os.path.join(BASE,'audio'); WORK=os.path.join(BASE,'extra_'+MODEL); os.makedirs(WORK,exist_ok=True)
SR=22050 if MODEL=='timbre' else 16000
CLEAN=os.path.join(AUDIO,'client_original_16k.wav')
TIMBRE_REPO=os.path.join(BASE,'TimbreWatermarking','watermarking_model')
if MODEL=='audioseal':
    WM=os.path.join(AUDIO,'native_wm_full.wav'); EMBEDDED='0110111111100100'
elif MODEL=='aware':
    WM=os.path.join(AUDIO,'aware_wm.wav'); EMBEDDED=open(os.path.join(AUDIO,'aware_bits.txt')).read().strip()
else:
    WM=os.path.join(BASE,'timbre_io','out','wmed-0','wavs','client.wav')
    CLEAN=os.path.join(BASE,'timbre_io','in','client.wav')
    EMBEDDED=''.join(map(str, eval(open(os.path.join(TIMBRE_REPO,'results','wmpool.txt')).read().strip().splitlines()[0])))
NB=len(EMBEDDED); _det=None
def _init():
    global _det
    import torch; torch.set_num_threads(1)
    if MODEL=='audioseal':
        from audioseal import AudioSeal; _det=AudioSeal.load_detector('audioseal_detector_16bits'); _det.eval()
    elif MODEL=='aware':
        from aware.utils.models import load; _,_det=load(name='AWARE')
    else:
        import yaml
        os.chdir(TIMBRE_REPO)
        sys.path.insert(0,TIMBRE_REPO); sys.path.insert(0,os.path.join(TIMBRE_REPO,'distortions'))
        from model.conv2_mel_modules import Decoder
        pc=yaml.load(open('config/process.yaml'),Loader=yaml.FullLoader)
        mc=yaml.load(open('config/model.yaml'),Loader=yaml.FullLoader)
        tc=yaml.load(open('config/train.yaml'),Loader=yaml.FullLoader)
        dec=Decoder(pc,mc,tc["watermark"]["length"],pc["audio"]["win_len"],mc["dim"]["embedding"],nlayers_decoder=mc["layer"]["nlayers_decoder"],attention_heads=mc["layer"]["attention_heads_decoder"])
        dev='cuda' if torch.cuda.is_available() else 'cpu'; dec=dec.to(dev)
        ckd='results/ckpt/pth'; ck=sorted(os.listdir(ckd),key=lambda x:os.path.getmtime(os.path.join(ckd,x)))[-1]
        m=torch.load(os.path.join(ckd,ck),map_location=dev); dec.load_state_dict(m["decoder"],strict=False)
        dec.eval(); dec.robust=False; _det=(dec,dev)
def _load(path):
    import soundfile as sf
    s,sr=sf.read(path)
    if getattr(s,'ndim',1)>1: s=s.mean(axis=1)
    return s.astype('float32')
def _detect(path):
    import numpy as np, torch
    if MODEL=='timbre':
        import librosa
        dec,dev=_det; y,_=librosa.load(path,sr=SR)
        x=torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad(): d=dec.test_forward(x)
        pred=(d.squeeze().detach().cpu().numpy().ravel()[:NB]>=0).astype(int)
        bits=''.join(map(str,pred.tolist())); acc=sum(a==b for a,b in zip(bits,EMBEDDED))/NB
        return acc, acc
    s=_load(path)
    if MODEL=='audioseal':
        x=torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        with torch.no_grad(): prob,msg=_det.detect_watermark(x,sample_rate=SR)
        bits=''.join(map(str,(msg.squeeze()>0.5).int().tolist())); conf=float(prob)
    else:
        from aware.service import detect_watermark
        b,c=detect_watermark(s,SR,_det); bits=''.join(map(str,np.array(b).astype(int).ravel()[:NB].tolist())); conf=float(c)
    return conf, sum(a==b for a,b in zip(bits,EMBEDDED))/NB
def _pesq(path):
    try:
        from pesq import pesq
        if MODEL=='timbre':
            import librosa; ref=librosa.load(CLEAN,sr=16000)[0]; deg=librosa.load(path,sr=16000)[0]
        else:
            ref=_load(CLEAN); deg=_load(path)
        n=min(len(ref),len(deg)); return float(pesq(16000,ref[:n],deg[:n],'wb'))
    except Exception: return None
def _ff(out,af=None):
    cmd=['ffmpeg','-y','-loglevel','error','-i',WM]
    if af: cmd+=['-af',af]
    cmd+=['-ar',str(SR),'-ac','1',out]; subprocess.run(cmd,check=True); return out
def _seg(name,fn):
    import soundfile as sf; w=_load(WM); out=os.path.join(WORK,name); sf.write(out, fn(w).astype('float32'), SR, subtype='PCM_16'); return out
def _noise(name, snr_db):
    import soundfile as sf; w=_load(WM); rms=float(np.sqrt((w**2).mean())); npow=rms/(10**(snr_db/20.0))
    n=np.random.RandomState(0).randn(len(w)).astype('float32')*npow
    out=os.path.join(WORK,name); sf.write(out,(w+n).astype('float32'),SR,subtype='PCM_16'); return out
ATTACKS={
 'tempo_up_10':   (lambda: _ff(os.path.join(WORK,'tu.wav'),'atempo=1.1'), False),
 'tempo_down_10': (lambda: _ff(os.path.join(WORK,'td.wav'),'atempo=0.9'), False),
 'crop_10s':      (lambda: _seg('c10.wav', lambda w: w[40*SR:50*SR]), False),
 'crop_5s':       (lambda: _seg('c5.wav', lambda w: w[40*SR:45*SR]), False),
 'crop_3s':       (lambda: _seg('c3.wav', lambda w: w[40*SR:43*SR]), False),
 'denoise_afftdn':(lambda: _ff(os.path.join(WORK,'dn.wav'),'afftdn=nr=20'), True),
 'noise_10db':    (lambda: _noise('n10.wav',10), True),
 'noise_5db':     (lambda: _noise('n5.wav',5), True),
}
ORDER=['tempo_up_10','tempo_down_10','crop_10s','crop_5s','crop_3s','denoise_afftdn','noise_10db','noise_5db']
def work(name):
    try:
        fn,pesq=ATTACKS[name]; path=fn(); conf,acc=_detect(path)
        pe=_pesq(path) if pesq else None; pe=round(pe,2) if (pe is not None and pe==pe) else None
        return (name,round(conf,3),round(acc,3),pe)
    except Exception as e:
        return (name,'ERR','ERR',str(e)[:60])
if __name__=='__main__':
    nw=1 if MODEL=='timbre' else min(len(ORDER), int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    print('MODEL=%s | embedded=%s | workers=%d'%(MODEL,EMBEDDED,nw), flush=True)
    with ProcessPoolExecutor(max_workers=nw, initializer=_init) as ex:
        results=list(ex.map(work, ORDER))
    print('\n==== EXTRA TESTS (%s) ===='%MODEL)
    print(f'{"test":16s} {"conf":>8s} {"bit_acc":>8s} {"pesq":>6s}')
    for r in results: print(f'{r[0]:16s} {str(r[1]):>8s} {str(r[2]):>8s} {str(r[3]):>6s}')
    with open(os.path.join(BASE,'extra_%s.csv'%MODEL),'w',newline='') as f:
        csv.writer(f).writerows([['test','conf','bit_accuracy','pesq']]+results)
    print('saved extra_%s.csv'%MODEL)

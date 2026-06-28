import os, sys, subprocess, math, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
BASE=os.path.expanduser('~/wm_compare'); AUDIO=os.path.join(BASE,'audio'); WORK=os.path.join(BASE,'attacked_aware'); os.makedirs(WORK,exist_ok=True)
SR=16000; WM=os.path.join(AUDIO,'aware_wm.wav'); CLEAN=os.path.join(AUDIO,'client_original_16k.wav')
EMBEDDED=open(os.path.join(AUDIO,'aware_bits.txt')).read().strip(); NB=len(EMBEDDED)
_det=None
def _init():
    global _det
    import torch; torch.set_num_threads(1)
    from aware.utils.models import load
    _,_det=load(name='AWARE')
def _load(path):
    import librosa; s,_=librosa.load(path, sr=SR, mono=True); return s.astype('float32')
def _detect(path):
    from aware.service import detect_watermark
    bits,conf=detect_watermark(_load(path), SR, _det)
    bits=''.join(map(str, np.array(bits).astype(int).ravel()[:NB].tolist()))
    return float(conf), sum(a==b for a,b in zip(bits,EMBEDDED))/NB
def _pesq(path):
    try:
        from pesq import pesq; ref=_load(CLEAN); deg=_load(path); n=min(len(ref),len(deg))
        return float(pesq(SR, ref[:n], deg[:n], 'wb'))
    except Exception: return None
def _ff(out,af=None):
    cmd=['ffmpeg','-y','-loglevel','error','-i',WM]
    if af: cmd+=['-af',af]
    cmd+=['-ar',str(SR),'-ac','1',out]; subprocess.run(cmd,check=True); return out
def _codec(tag,args):
    enc=os.path.join(WORK,tag); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',WM]+args+[enc],check=True)
    out=enc+'.wav'; subprocess.run(['ffmpeg','-y','-loglevel','error','-i',enc,'-ar',str(SR),'-ac','1',out],check=True); return out
def _chain():
    a=os.path.join(WORK,'ch.m4a'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',WM,'-c:a','aac','-b:a','128k',a],check=True)
    m=os.path.join(WORK,'ch.mp3'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',a,'-codec:a','libmp3lame','-b:a','128k',m],check=True)
    w=os.path.join(WORK,'ch.wav'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',m,'-ar',str(SR),'-ac','1',w],check=True); return w
def _seg(name,fn):
    import soundfile as sf; w=_load(WM); out=os.path.join(WORK,name); sf.write(out, fn(w).astype('float32'), SR, subtype='PCM_16'); return out
ATTACKS={
 '00_baseline': (lambda: WM, True),
 'mp3_64k': (lambda: _codec('c64.mp3',['-codec:a','libmp3lame','-b:a','64k']), True),
 'mp3_128k': (lambda: _codec('c128.mp3',['-codec:a','libmp3lame','-b:a','128k']), True),
 'mp3_320k': (lambda: _codec('c320.mp3',['-codec:a','libmp3lame','-b:a','320k']), True),
 'format_chain': (_chain, True),
 'edit_trim_head5s': (lambda: _seg('trim.wav', lambda w: w[5*SR:]), False),
 'edit_add_silence2s': (lambda: _seg('sil.wav', lambda w: np.concatenate([np.zeros(2*SR,'float32'),w])), False),
 'edit_splice_cut5s': (lambda: _seg('spl.wav', lambda w: np.concatenate([w[:40*SR],w[45*SR:]])), False),
 'sig_eq': (lambda: _ff(os.path.join(WORK,'eq.wav'),'bass=g=-6,treble=g=6'), True),
 'sig_pitch_up': (lambda: _ff(os.path.join(WORK,'pit.wav'),'asetrate=16944,aresample=16000,atempo=0.9443'), False),
 'sig_normalize': (lambda: _ff(os.path.join(WORK,'nrm.wav'),'loudnorm'), True),
 'platform_opus64': (lambda: _codec('p.opus',['-c:a','libopus','-b:a','64k']), True),
 'platform_aac96': (lambda: _codec('pa.m4a',['-c:a','aac','-b:a','96k']), True),
 'rerecord_sim': (lambda: _ff(os.path.join(WORK,'rr.wav'),'aecho=0.8:0.9:60:0.3,highpass=f=80,lowpass=f=7000'), False),
}
ORDER=['00_baseline','mp3_64k','mp3_128k','mp3_320k','format_chain','edit_trim_head5s','edit_add_silence2s','edit_splice_cut5s','sig_eq','sig_pitch_up','sig_normalize','platform_opus64','platform_aac96','rerecord_sim']
def work(name):
    try:
        fn,pesq=ATTACKS[name]; path=fn(); conf,acc=_detect(path)
        pe=_pesq(path) if pesq else None; pe=round(pe,2) if (pe is not None and pe==pe) else None
        return (name,round(conf,3),round(acc,3),pe)
    except Exception as e:
        return (name,'ERR','ERR',str(e)[:50])
if __name__=='__main__':
    nw=min(len(ORDER), int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    print('workers:',nw,'| embedded:',EMBEDDED, flush=True)
    with ProcessPoolExecutor(max_workers=nw, initializer=_init) as ex:
        results=list(ex.map(work, ORDER))
    print('\n==== SUMMARY (AWARE / aware_wm) ====')
    print(f'{"test":24s} {"conf":>8s} {"bit_acc":>8s} {"pesq":>6s}')
    for r in results: print(f'{r[0]:24s} {str(r[1]):>8s} {str(r[2]):>8s} {str(r[3]):>6s}')
    with open(os.path.join(BASE,'babar_aware_results.csv'),'w',newline='') as f:
        csv.writer(f).writerows([['test','confidence','bit_accuracy','pesq']]+results)
    print('saved:', os.path.join(BASE,'babar_aware_results.csv'))

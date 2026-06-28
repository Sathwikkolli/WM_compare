import os, sys, subprocess, math, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
BASE=os.path.expanduser('~/wm_compare'); TOOLKIT=os.path.join(BASE,'Audio-Watermark-Toolkit')
AUDIO=os.path.join(BASE,'audio'); WORK=os.path.join(BASE,'attacked'); os.makedirs(WORK,exist_ok=True)
sys.path.insert(0,TOOLKIT)
SR=16000; EMBEDDED='0110111111100100'
CLEAN=os.path.join(AUDIO,'client_original_16k.wav'); WM=os.path.join(AUDIO,'native_wm_full.wav')
_det=None
def _init():
    global _det
    import torch; torch.set_num_threads(1)
    from audioseal import AudioSeal
    _det=AudioSeal.load_detector('audioseal_detector_16bits'); _det.eval()
def _detect(path):
    import torch
    from utils_audio import read_wav_mono
    wav,sr=read_wav_mono(path)
    if sr!=SR:
        from scipy.signal import resample_poly; g=math.gcd(sr,SR); wav=resample_poly(wav,SR//g,sr//g).astype('float32')
    x=torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    with torch.no_grad(): prob,msg=_det.detect_watermark(x,sample_rate=SR)
    bits=''.join(map(str,(msg.squeeze()>0.5).int().tolist()))
    return float(prob), sum(a==b for a,b in zip(bits,EMBEDDED))/len(EMBEDDED)
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
    from utils_audio import read_wav_mono, write_wav_pcm16
    w,_=read_wav_mono(WM); out=os.path.join(WORK,name); write_wav_pcm16(out,fn(w).astype('float32'),SR); return out
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
        from utils_pesq import pesq_pair_files
        fn,pesq=ATTACKS[name]
        path=fn(); prob,acc=_detect(path)
        pe=pesq_pair_files(CLEAN,path) if pesq else None
        pe=round(pe,2) if (pe is not None and pe==pe) else None
        return (name,round(prob,3),round(acc,3),pe)
    except Exception as e:
        return (name,'ERR','ERR',str(e)[:50])
if __name__=='__main__':
    nw=min(len(ORDER), int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    print('parallel workers:',nw, flush=True)
    with ProcessPoolExecutor(max_workers=nw, initializer=_init) as ex:
        results=list(ex.map(work, ORDER))
    print('\n==== SUMMARY (AudioSeal / native_wm_full) ====')
    print(f'{"test":24s} {"det_prob":>8s} {"bit_acc":>8s} {"pesq":>6s}')
    for r in results: print(f'{r[0]:24s} {str(r[1]):>8s} {str(r[2]):>8s} {str(r[3]):>6s}')
    with open(os.path.join(BASE,'babar_audioseal_results.csv'),'w',newline='') as f:
        csv.writer(f).writerows([['test','detection_prob','bit_accuracy','pesq']]+results)
    print('saved:', os.path.join(BASE,'babar_audioseal_results.csv'))

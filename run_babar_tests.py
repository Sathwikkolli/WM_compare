import os, sys, subprocess, math, csv
import numpy as np
BASE=os.path.expanduser('~/wm_compare'); TOOLKIT=os.path.join(BASE,'Audio-Watermark-Toolkit')
AUDIO=os.path.join(BASE,'audio'); WORK=os.path.join(BASE,'attacked'); os.makedirs(WORK,exist_ok=True)
sys.path.insert(0,TOOLKIT)
import torch
from utils_audio import read_wav_mono, write_wav_pcm16
from utils_pesq import pesq_pair_files
from audioseal import AudioSeal
SR=16000; EMBEDDED='0110111111100100'
CLEAN=os.path.join(AUDIO,'client_original_16k.wav'); WM=os.path.join(AUDIO,'native_wm_full.wav')
det=AudioSeal.load_detector('audioseal_detector_16bits'); det.eval()
def detect(path):
    wav,sr=read_wav_mono(path)
    if sr!=SR:
        from scipy.signal import resample_poly; g=math.gcd(sr,SR); wav=resample_poly(wav,SR//g,sr//g).astype('float32')
    x=torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
    with torch.no_grad(): prob,msg=det.detect_watermark(x,sample_rate=SR)
    bits=''.join(map(str,(msg.squeeze()>0.5).int().tolist()))
    acc=sum(a==b for a,b in zip(bits,EMBEDDED))/len(EMBEDDED)
    return float(prob),acc
def ff(out, af=None):
    cmd=['ffmpeg','-y','-loglevel','error','-i',WM]
    if af: cmd+=['-af',af]
    cmd+=['-ar',str(SR),'-ac','1',out]; subprocess.run(cmd,check=True); return out
def codec(tag, args):
    enc=os.path.join(WORK,tag); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',WM]+args+[enc],check=True)
    out=enc+'.wav'; subprocess.run(['ffmpeg','-y','-loglevel','error','-i',enc,'-ar',str(SR),'-ac','1',out],check=True); return out
def chain():
    a=os.path.join(WORK,'ch.m4a'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',WM,'-c:a','aac','-b:a','128k',a],check=True)
    m=os.path.join(WORK,'ch.mp3'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',a,'-codec:a','libmp3lame','-b:a','128k',m],check=True)
    w=os.path.join(WORK,'ch.wav'); subprocess.run(['ffmpeg','-y','-loglevel','error','-i',m,'-ar',str(SR),'-ac','1',w],check=True); return w
def trimh():
    w,_=read_wav_mono(WM); p=os.path.join(WORK,'trim.wav'); write_wav_pcm16(p,w[5*SR:].astype('float32'),SR); return p
def addsil():
    w,_=read_wav_mono(WM); p=os.path.join(WORK,'sil.wav'); write_wav_pcm16(p,np.concatenate([np.zeros(2*SR,'float32'),w]).astype('float32'),SR); return p
def splice():
    w,_=read_wav_mono(WM); p=os.path.join(WORK,'spl.wav'); write_wav_pcm16(p,np.concatenate([w[:40*SR],w[45*SR:]]).astype('float32'),SR); return p
results=[]
def run(name, fn, pesq=True):
    try:
        path=fn(); prob,acc=detect(path); pe=pesq_pair_files(CLEAN,path) if pesq else None
        pe=round(pe,2) if (pe is not None and pe==pe) else None
        results.append((name,round(prob,3),round(acc,3),pe)); print(f'{name:24s} prob={prob:.3f} bitacc={acc:.3f} pesq={pe}')
    except Exception as e:
        results.append((name,'ERR','ERR',None)); print(f'{name:24s} ERROR: {e}')
run('00_baseline', lambda: WM)
run('mp3_64k',  lambda: codec('c64.mp3',['-codec:a','libmp3lame','-b:a','64k']))
run('mp3_128k', lambda: codec('c128.mp3',['-codec:a','libmp3lame','-b:a','128k']))
run('mp3_320k', lambda: codec('c320.mp3',['-codec:a','libmp3lame','-b:a','320k']))
run('format_chain', chain)
run('edit_trim_head5s', trimh, pesq=False)
run('edit_add_silence2s', addsil, pesq=False)
run('edit_splice_cut5s', splice, pesq=False)
run('sig_eq', lambda: ff(os.path.join(WORK,'eq.wav'), af='bass=g=-6,treble=g=6'))
run('sig_pitch_up', lambda: ff(os.path.join(WORK,'pit.wav'), af='asetrate=16944,aresample=16000,atempo=0.9443'), pesq=False)
run('sig_normalize', lambda: ff(os.path.join(WORK,'nrm.wav'), af='loudnorm'))
run('platform_opus64', lambda: codec('p.opus',['-c:a','libopus','-b:a','64k']))
run('platform_aac96', lambda: codec('pa.m4a',['-c:a','aac','-b:a','96k']))
run('rerecord_sim', lambda: ff(os.path.join(WORK,'rr.wav'), af='aecho=0.8:0.9:60:0.3,highpass=f=80,lowpass=f=7000'), pesq=False)
print('\n==== SUMMARY (AudioSeal / native_wm_full) ====')
print(f'{"test":24s} {"det_prob":>8s} {"bit_acc":>8s} {"pesq":>6s}')
for n,pr,ac,pe in results: print(f'{n:24s} {str(pr):>8s} {str(ac):>8s} {str(pe):>6s}')
with open(os.path.join(BASE,'babar_audioseal_results.csv'),'w',newline='') as f:
    csv.writer(f).writerows([['test','detection_prob','bit_accuracy','pesq_wb']]+results)
print('saved:', os.path.join(BASE,'babar_audioseal_results.csv'))

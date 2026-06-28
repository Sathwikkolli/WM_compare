import os, subprocess, csv, numpy as np, torch, librosa, soundfile as sf, yaml

BASE=os.path.expanduser('~/wm_compare')
REPO=os.path.join(BASE,'TimbreWatermarking','watermarking_model')
WORK=os.path.join(BASE,'bench_timbre'); os.makedirs(WORK,exist_ok=True)
SR=22050; WM_INDEX=0
CLEAN=os.path.join(BASE,'timbre_io','in','client.wav')                  # watermark-free (resampled orig)
WM=os.path.join(BASE,'timbre_io','out','wmed-%d'%WM_INDEX,'wavs','client.wav')  # timbre-watermarked
CKPT_DIR=os.path.join(REPO,'results','ckpt','pth')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

# ---- ground-truth message (wmpool line WM_INDEX) ----
wmpool=open(os.path.join(REPO,'results','wmpool.txt')).read().strip().splitlines()
WM_BITS=eval(wmpool[WM_INDEX])                      # e.g. [1,1,1,1,0,0,1,1,1,0]
EMBEDDED=''.join(map(str,WM_BITS)); NB=len(WM_BITS)

# ---- build Timbre decoder exactly like extract.py ----
def _load_cfg(p): return yaml.load(open(p), Loader=yaml.FullLoader)
pc=_load_cfg(os.path.join(REPO,'config','process.yaml'))
mc=_load_cfg(os.path.join(REPO,'config','model.yaml'))
tc=_load_cfg(os.path.join(REPO,'config','train.yaml'))
import sys; sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,'distortions'))
from model.conv2_mel_modules import Decoder
win_dim=pc["audio"]["win_len"]; emb=mc["dim"]["embedding"]
nld=mc["layer"]["nlayers_decoder"]; ahd=mc["layer"]["attention_heads_decoder"]
msg_len=tc["watermark"]["length"]
decoder=Decoder(pc,mc,msg_len,win_dim,emb,nlayers_decoder=nld,attention_heads=ahd).to(DEVICE)
ck=sorted(os.listdir(CKPT_DIR), key=lambda x:os.path.getmtime(os.path.join(CKPT_DIR,x)))[-1]
model=torch.load(os.path.join(CKPT_DIR,ck), map_location=DEVICE)
decoder.load_state_dict(model["decoder"], strict=False); decoder.eval(); decoder.robust=False
print('loaded decoder from',ck,'| embedded=',EMBEDDED,'| device',DEVICE, flush=True)

def _detect(path):
    y,_=librosa.load(path, sr=SR)                  # mono float32
    wav=torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,N]
    with torch.no_grad(): decoded=decoder.test_forward(wav)
    pred=(decoded.squeeze().detach().cpu().numpy().ravel()[:NB] >= 0).astype(int)
    bit_acc=float((pred==np.array(WM_BITS)).mean())
    return ''.join(map(str,pred.tolist())), bit_acc

def _pesq(deg):
    try:
        from pesq import pesq
        r=librosa.load(CLEAN,sr=16000)[0]; d=librosa.load(deg,sr=16000)[0]
        n=min(len(r),len(d)); v=pesq(16000,r[:n],d[:n],'wb'); return round(float(v),2)
    except Exception: return None

# ---- attacks (mirror run_babar_tests_par.py, at SR=22050) ----
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
    w,_=librosa.load(WM,sr=SR); out=os.path.join(WORK,name); sf.write(out,fn(w).astype('float32'),SR,subtype='PCM_16'); return out

ATTACKS={
 '00_baseline':       (lambda: WM, True),
 'mp3_64k':           (lambda: _codec('c64.mp3',['-codec:a','libmp3lame','-b:a','64k']), True),
 'mp3_128k':          (lambda: _codec('c128.mp3',['-codec:a','libmp3lame','-b:a','128k']), True),
 'mp3_320k':          (lambda: _codec('c320.mp3',['-codec:a','libmp3lame','-b:a','320k']), True),
 'format_chain':      (_chain, True),
 'edit_trim_head5s':  (lambda: _seg('trim.wav', lambda w: w[5*SR:]), False),
 'edit_add_silence2s':(lambda: _seg('sil.wav', lambda w: np.concatenate([np.zeros(2*SR,'float32'),w])), False),
 'edit_splice_cut5s': (lambda: _seg('spl.wav', lambda w: np.concatenate([w[:40*SR],w[45*SR:]])), False),
 'sig_eq':            (lambda: _ff(os.path.join(WORK,'eq.wav'),'bass=g=-6,treble=g=6'), True),
 'sig_pitch_up':      (lambda: _ff(os.path.join(WORK,'pit.wav'),'asetrate=23351,aresample=22050,atempo=0.9443'), False),
 'sig_normalize':     (lambda: _ff(os.path.join(WORK,'nrm.wav'),'loudnorm'), True),
 'platform_opus64':   (lambda: _codec('p.opus',['-c:a','libopus','-b:a','64k']), True),
 'platform_aac96':    (lambda: _codec('pa.m4a',['-c:a','aac','-b:a','96k']), True),
 'rerecord_sim':      (lambda: _ff(os.path.join(WORK,'rr.wav'),'aecho=0.8:0.9:60:0.3,highpass=f=80,lowpass=f=7000'), False),
}
ORDER=['00_baseline','mp3_64k','mp3_128k','mp3_320k','format_chain','edit_trim_head5s','edit_add_silence2s','edit_splice_cut5s','sig_eq','sig_pitch_up','sig_normalize','platform_opus64','platform_aac96','rerecord_sim']

rows=[]
for name in ORDER:
    try:
        fn,do_pesq=ATTACKS[name]; path=fn()
        bits,acc=_detect(path); pe=_pesq(path) if do_pesq else None
        det='Detected' if acc>=0.8 else 'Not Detected'
        rows.append((name,det,round(acc,3),bits,pe))
    except Exception as e:
        rows.append((name,'ERR','ERR','',str(e)[:60]))

print('\n==== TIMBRE BENCHMARK (Detected = bit_acc >= 0.8) ====')
print(f'{"test":20s} {"result":12s} {"bit_acc":>8s} {"bits":>12s} {"pesq":>6s}')
for r in rows: print(f'{r[0]:20s} {r[1]:12s} {str(r[2]):>8s} {str(r[3]):>12s} {str(r[4]):>6s}')
with open(os.path.join(BASE,'bench_timbre.csv'),'w',newline='') as f:
    csv.writer(f).writerows([['test','result','bit_accuracy','decoded_bits','pesq']]+rows)
print('\nsaved bench_timbre.csv')

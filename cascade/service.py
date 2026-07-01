"""
service.py  --  deployment-facing API for the 3-watermark system.

Two functions the web backend calls; everything else (model loading, resampling,
cascading, per-user keys) is handled internally.

    embed(in_path, out_path, user_id, models=('audioseal','aware','timbre'))
        Embed the SAME user_id into each requested watermark, cascaded into one
        audio file. `models` is any subset -> pick one, two, or all three.

    detect(path)
        Run every detector, recover the user_id from each, and report a
        consensus. Works whether the file carries one watermark or all three.

Setup on the deployment box
    export WM_COMPARE_BASE=/path/to/wm_compare      # where the repos + weights live
    # requires: audioseal (pip), the aware repo on sys.path, the Timbre repo +
    # its results/ckpt/pth/*.pth.tar checkpoint. See HANDOFF.md.

Design notes
    * user_id is 0 .. 1023 (Timbre's 10-bit payload is the ceiling). The same id
      goes into all three models via user_key.make_keys(); any single detector
      can recover it, giving 3x redundancy.
    * All audio is processed at 22.05 kHz (SR_MASTER); adapters resample to each
      model's native rate internally.
    * Models are loaded lazily and cached by get_adapter(), so the FIRST call is
      slow (weights load; AWARE also runs a 400-iter optimisation per embed) and
      later calls reuse the warm models. In a server, call warmup() at startup.

CLI
    python service.py embed  in.wav out.wav 42
    python service.py embed  in.wav out.wav 42 --models audioseal,timbre
    python service.py detect out.wav
"""

import os
import cascade_lib as cl
import user_key as uk

MODELS = ('audioseal', 'aware', 'timbre')


def _set_message(adapter, bits):
    """Point an adapter at a specific message before embedding."""
    adapter.truth = bits
    if adapter.code == 'timbre':                 # Timbre embeds from _wm (int list)
        adapter._wm = [int(b) for b in bits]
        adapter.NB  = len(bits)


def warmup(models=MODELS):
    """Load + cache the requested models. Call once at server startup."""
    for m in models:
        cl.get_adapter(m)


def embed(in_path, out_path, user_id, models=MODELS):
    """Cascade-embed `user_id` into `in_path`, write to `out_path`.

    Returns a dict with the output path, the user id, and the per-model bit
    string that was embedded.
    """
    models = tuple(models)
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; choose from {MODELS}")
    keys = uk.make_keys(user_id)                 # same id -> per-model payloads

    y = cl.read_wav(in_path, sr=cl.SR_MASTER)    # mono float32 @ 22.05k
    for m in models:                             # cascade: each embed writes over the last
        adapter = cl.get_adapter(m)
        _set_message(adapter, keys[m])
        y = adapter.embed(y)
    cl.write_wav(out_path, y, sr=cl.SR_MASTER)
    return {'path': out_path, 'user_id': user_id,
            'models': list(models), 'keys': {m: keys[m] for m in models}}


def detect(path):
    """Run all three detectors on `path` and recover the user id from each.

    Returns per-model results plus a consensus. A model is considered to carry a
    watermark when its bit-accuracy vs the recovered id is high; here we simply
    report each model's recovered id + confidence and a majority-vote consensus.
    """
    y = cl.read_wav(path, sr=cl.SR_MASTER)
    per_model = {}
    for m in MODELS:
        adapter = cl.get_adapter(m)
        conf, bits, _ = adapter.detect(y)
        per_model[m] = {'user_id': uk.recover_id(bits),
                        'confidence': round(float(conf), 4),
                        'bits': bits}
    # consensus = most common recovered id across the three detectors
    ids = [r['user_id'] for r in per_model.values()]
    consensus = max(set(ids), key=ids.count)
    agree = ids.count(consensus)
    return {'consensus_user_id': consensus,
            'agreement': f'{agree}/{len(MODELS)}',
            'per_model': per_model}


def _cli(argv):
    if not argv:
        print("usage: python service.py embed <in> <out> <user_id> [--models a,b,c]")
        print("       python service.py detect <path>")
        return 1
    cmd = argv[0]
    if cmd == 'embed':
        in_path, out_path, uid = argv[1], argv[2], int(argv[3])
        models = MODELS
        if '--models' in argv:
            models = tuple(argv[argv.index('--models') + 1].split(','))
        r = embed(in_path, out_path, uid, models)
        print(f"embedded user {r['user_id']} via {r['models']} -> {r['path']}")
        for m, b in r['keys'].items():
            print(f"  {m:9s}: {b}")
    elif cmd == 'detect':
        r = detect(argv[1])
        print(f"consensus user_id: {r['consensus_user_id']}  (agreement {r['agreement']})")
        for m, d in r['per_model'].items():
            print(f"  {m:9s}: id={d['user_id']:<4d} conf={d['confidence']:<8} bits={d['bits']}")
    else:
        print(f"unknown command {cmd!r}"); return 1
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_cli(sys.argv[1:]))

"""
user_key.py  --  turn a user ID into the per-model watermark messages.

The cascade embeds three watermarks (AudioSeal 16-bit, AWARE 20-bit, Timbre
10-bit) into one audio file. To tell users apart we embed the SAME user ID in
all three. The smallest payload (Timbre = 10 bits) sets the ceiling:

        10 bits  ->  2**10  =  1024 distinguishable users  (IDs 0 .. 1023).

The 10-bit ID is the canonical key. For the wider payloads (AudioSeal 16,
AWARE 20) we TILE the 10 ID bits cyclically to fill the extra room, so every
model carries the whole ID plus some redundancy. On the way back, each model's
detected bits are folded back onto the 10 positions and majority-voted, which
also repairs a few attack-flipped bits for free.

    embed :  make_keys(uid)   -> {'audioseal','aware','timbre': bitstring}
    detect:  recover_id(bits) -> uid                        (per model)

CLI
    python user_key.py 42                 # show the three keys for user 42
    python user_key.py --user alice@x.com # hash a username -> id -> keys
    python user_key.py --decode 0000101010 0000101010...    # bits -> id
"""

import hashlib

# Payload capacity of each model, in bits (must match cascade_lib adapters).
CAPACITY = {'timbre': 10, 'audioseal': 16, 'aware': 20}

CANON_BITS = 10                 # Timbre is the bottleneck -> canonical id width
MAX_USERS  = 1 << CANON_BITS    # 1024


def id_to_bits(user_id, nbits=CANON_BITS):
    """Integer id -> fixed-width MSB-first bit string, e.g. 5 -> '0000000101'."""
    if not (0 <= user_id < (1 << nbits)):
        raise ValueError(f"user_id {user_id} out of range for {nbits} bits "
                         f"(valid 0 .. {(1 << nbits) - 1})")
    return format(user_id, f'0{nbits}b')


def _tile(base, n):
    """Repeat `base` cyclically to length n: '0101',6 -> '010101'."""
    return ''.join(base[i % len(base)] for i in range(n))


def make_keys(user_id):
    """user id (0 .. 1023) -> {model: bitstring} ready to embed.

    Timbre gets the raw 10-bit id; AudioSeal/AWARE get the id tiled to their
    width so every watermark encodes the same user with extra redundancy.
    """
    base = id_to_bits(user_id, CANON_BITS)
    return {model: _tile(base, width) for model, width in CAPACITY.items()}


def recover_id(detected_bits, canon=CANON_BITS):
    """Detected bits from ONE model -> recovered user id.

    Folds the (possibly tiled) payload back onto `canon` positions and takes a
    majority vote per position. Ties fall back to the first (primary) copy.
    Works for any model: pass whatever bit string that detector returned.
    """
    bits = ''.join('1' if b in (1, '1', True) else '0' for b in detected_bits)
    if len(bits) < canon:
        raise ValueError(f"need at least {canon} bits, got {len(bits)}")
    groups = [[] for _ in range(canon)]
    for i, ch in enumerate(bits):
        groups[i % canon].append(int(ch))
    out = []
    for v in groups:
        ones, zeros = sum(v), len(v) - sum(v)
        out.append('1' if ones > zeros else '0' if zeros > ones else str(v[0]))
    return int(''.join(out), 2)


def username_to_id(username):
    """Deterministically map an arbitrary username/email to a 0..1023 id.

    Convenience only. WARNING: 1024 slots means hash COLLISIONS become likely
    after ~40 users (birthday bound ~= 1.2*sqrt(1024)). For guaranteed
    uniqueness, assign sequential ids from a database instead of hashing.
    """
    h = hashlib.sha256(username.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') % MAX_USERS


def _cli(argv):
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("usage: python user_key.py <id> | --user <name> | --decode <bits...>")
        return 1
    if argv[0] == '--decode':
        for bits in argv[1:]:
            print(f"{bits}  ->  user_id {recover_id(bits)}")
        return 0
    if argv[0] == '--user':
        uid = username_to_id(argv[1])
        print(f"username {argv[1]!r}  ->  user_id {uid}  (hashed; watch for collisions)")
    else:
        uid = int(argv[0])
    keys = make_keys(uid)
    print(f"user_id {uid}   (canonical 10-bit key: {id_to_bits(uid)})")
    for model in ('audioseal', 'aware', 'timbre'):
        print(f"  {model:9s} {CAPACITY[model]:2d} bits : {keys[model]}")
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_cli(sys.argv[1:]))

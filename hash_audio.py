#!/usr/bin/env python3
"""
hash_audio.py  --  SHA-256 integrity hashing for audio files (FakeXpose).

Fingerprints the EXACT BYTES of a file. This is the integrity/chain-of-custody
layer, completely separate from watermarking:
    * hash  -> proves the file bytes are unchanged since we saw them
    * watermark (service.py) -> proves who the file came from

Stdlib only -- no pip installs, runs as-is on Great Lakes.

USAGE
    python hash_audio.py <audio_path>
    python hash_audio.py <audio_path> --record custody.jsonl     # append a custody record
    python hash_audio.py <audio_path> --verify 3c2c99e0...       # check against a known hash
    python hash_audio.py --demo <audio_path>                     # show the avalanche effect

NOTES
    * Hashes the raw file bytes, NOT decoded audio. Re-encoding the same
      recording produces a totally different hash -- that is correct and expected.
    * SHA-256 is the NIST CFTT forensic default. MD5 is computed only as a
      supplementary cross-check (it is collision-broken -- never the sole proof).
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

CHUNK = 1 << 16  # 64 KiB read blocks (result is identical to hashing in one shot)


def hash_file(path, algo="sha256"):
    """Return the hex digest of a file's raw bytes using `algo`."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:                      # "rb" = raw bytes, no decoding
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_record(path):
    """Compute the full integrity record for a file."""
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "size_bytes": st.st_size,
        "sha256": hash_file(path, "sha256"),         # primary / legal anchor
        "md5": hash_file(path, "md5"),               # supplementary cross-check only
        "hashed_at_utc": utc_now(),
        "operator": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
    }


def print_record(rec):
    print(f"file          : {rec['filename']}")
    print(f"size (bytes)  : {rec['size_bytes']:,}")
    print(f"sha256        : {rec['sha256']}")
    print(f"md5 (suppl.)  : {rec['md5']}")
    print(f"hashed_at_utc : {rec['hashed_at_utc']}")
    print(f"operator      : {rec['operator']}")


def demo_avalanche(path):
    """Flip ONE bit and show ~half the hash output changes."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    original = hashlib.sha256(data).hexdigest()

    i = min(5000, len(data) - 1)                     # pick a byte in the middle
    data[i] ^= 0x01                                  # flip the lowest bit
    flipped = hashlib.sha256(data).hexdigest()

    # count differing bits between the two 256-bit digests
    a = int(original, 16)
    b = int(flipped, 16)
    diff_bits = bin(a ^ b).count("1")

    print("--- avalanche demo (same file, one bit flipped) ---")
    print(f"byte flipped  : index {i}  (file size unchanged: {len(data):,} bytes)")
    print(f"ORIGINAL      : {original}")
    print(f"1-BIT-CHANGED : {flipped}")
    print(f"=> {diff_bits} of 256 output bits changed (~{diff_bits/256*100:.0f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="SHA-256 integrity hashing for audio files.")
    ap.add_argument("audio_path", help="path to the audio file to hash")
    ap.add_argument("--record", metavar="FILE",
                    help="append the integrity record as one JSON line to FILE (append-only custody log)")
    ap.add_argument("--verify", metavar="HASH",
                    help="compare the file's SHA-256 against a known hash and report MATCH/MISMATCH")
    ap.add_argument("--demo", action="store_true",
                    help="also show the avalanche effect (one flipped bit)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.audio_path):
        print(f"error: not a file: {args.audio_path}", file=sys.stderr)
        return 2

    rec = build_record(args.audio_path)
    print_record(rec)

    if args.verify:
        expected = args.verify.strip().lower()
        actual = rec["sha256"].lower()
        if actual == expected:
            print("verify        : MATCH  (bytes unchanged)")
        else:
            print("verify        : MISMATCH  (file has changed!)")
            print(f"                expected {expected}")
            print(f"                actual   {actual}")

    if args.record:
        with open(args.record, "a", encoding="utf-8") as f:   # append-only = custody log
            f.write(json.dumps(rec) + "\n")
        print(f"record        : appended to {args.record}")

    if args.demo:
        print()
        demo_avalanche(args.audio_path)

    # exit non-zero on a verify mismatch so it is scriptable in a pipeline
    if args.verify and rec["sha256"].lower() != args.verify.strip().lower():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Cryptographic Hashing for Audio Integrity — Briefing Notes

*Speaking notes for presenting the hashing / tamper-proofing work. Structured as
"what to say" in order, with the proof points and sources to cite. Not slides —
talk from this.*

---

## 0. The one-line thesis (open with this)

> "To prove a piece of audio hasn't been tampered with, you attach a cryptographic
> hash — a fixed-size fingerprint of the exact bytes. If any single bit changes,
> the fingerprint changes completely. For court/evidence audio the field-standard
> algorithm is **SHA-256**, wrapped in a documented chain of custody. But a hash
> only proves integrity *from the moment we receive the file* — proving the
> recording wasn't edited *before* it reached us is a separate forensic problem."

That sentence contains the whole talk. Everything below expands it.

---

## 1. Frame the problem: three things people call "hashing audio"

Make this distinction early — it shows you understand the landscape, and it's the
#1 thing people conflate.

| Technique | Fingerprints… | Changes when… | Used for |
|---|---|---|---|
| **Cryptographic hash** (SHA-256, SHA-3, BLAKE3) | the exact **bytes** | any 1 bit changes | integrity, tamper-proof, signatures |
| **Perceptual / acoustic hash** (Shazam, Chromaprint) | what it **sounds like** | the *sound* changes | content ID, dedup, "same song?" |
| **Watermark** (our AudioSeal / AWARE / Timbre work) | data hidden **inside** the signal | (designed to survive attacks) | provenance, per-user tracing |

Key cited fact: *"Acoustic fingerprints are **not** cryptographic hash functions"* —
a crypto hash tolerates zero change; a perceptual hash tolerates changes inaudible
to humans. (Source: Wikipedia — Acoustic fingerprint.)

**Our project = the third column (watermarking). Hashing = the first column.**
They are complementary: the hash proves *the file is intact*; the watermark proves
*where it came from*.

---

## 2. What a cryptographic hash IS (explain, then demo)

A one-way function: **any-size input → fixed-size fingerprint.** No key, not
reversible, not compression. Five properties to state:

1. **Deterministic** — same bytes always give the same hash.
2. **Fixed size** — a 1.8 MB MP3 and a 4 GB video both produce 64 hex chars (SHA-256).
3. **Avalanche** — change one bit, ~half the output bits flip. No "close."
4. **One-way (preimage resistance)** — can't reverse the fingerprint back to the file.
5. **Collision resistance** — can't find two different files with the same fingerprint.
   *This is the property that makes it legal "proof."*

### Live demo to run in front of him (reproducible on our own repo file)

Run on `attacked/c128.mp3`. These are the actual results:

```
File size:           1,864,556 bytes
SHA-256:             3c2c99e0f162819f8f2cb5e7e3500e210423b251387f8650c5f2e783b14a4075
Run it again:        3c2c99e0...  (identical — deterministic)

Flip ONE bit (byte 5000: 85 -> 84, file still 1,864,556 bytes):
ORIGINAL   SHA-256:  3c2c99e0f162819f8f2cb5e7e3500e21...
1-BIT-OFF  SHA-256:  243e0f7e69293265713337591082a737...
=> 136 of 256 output bits changed (~50%) — the avalanche effect, proven.
```

Command to reproduce live: `sha256sum attacked/c128.mp3`

**The line to say:** "One flipped bit out of ~15 million, and half the fingerprint
changes. That's why it's called *tamper-evident* — you cannot nudge the file
without the fingerprint screaming."

---

## 3. Why "exact bytes" matters (the crucial catch)

The hash is over raw bytes, not sound. State both sides honestly:

- ✅ **Perfect for integrity** — corruption, a swapped file, a malicious edit all
  change at least one byte, so the hash won't match. This is how software
  downloads, Git commits, and blockchain blocks verify themselves.
- ⚠️ **Says nothing about audio content** — re-encode the same recording to a new
  format and *every byte changes*, so the hash is totally different even though it
  sounds identical. A crypto hash answers "is this the same *file*?" — NOT "is this
  the same *recording*?"

---

## 4. The court / evidence use case (the core of the ask)

### 4a. Two meanings of "tampered" — we need both

| | **Byte integrity** (hash proves) | **Audio authenticity** (hash CANNOT prove) |
|---|---|---|
| Question | "Changed since *we* received it?" | "Edited/spliced *before* it reached us?" |
| Tool | SHA-256 + chain of custody | Forensic audio authentication exam |
| Standard | NIST CFTT, ISO 27037 | SWGDE Digital Audio Authentication |

**Say this explicitly:** a hash is a seal from our custody point *forward*. If the
audio was edited before we got it, its SHA-256 is a valid hash *of the edited file*.
So evidence work needs both — SHA-256 for chain of custody, and (if authenticity is
challenged) a SWGDE audio-authentication analysis (spectrogram for splices/aliasing/
filtering, ENF power-line analysis). Hashing can't do that second part.

### 4b. Which hash — and why SHA-256

- **NIST CFTT (Computer Forensics Tool Testing) sets SHA-256 as the default** hash
  for forensic imaging. Choosing a NIST-validated algorithm matters for
  admissibility (Daubert / Federal Rule of Evidence 702) — you can point to
  independent validation.
- Standard tools (FTK Imager, EnCase) auto-compute **MD5 + SHA-1 + SHA-256** on
  acquisition and re-verify the copy against the source.
- **Dual-hashing** (SHA-256 + MD5/SHA-1) is common practice — for cross-checking
  and backward compatibility with older case systems. But MD5/SHA-1 are broken for
  *collisions*, so: **SHA-256 is the legal anchor; MD5/SHA-1 are supplementary,
  never the sole proof.**
- Don't use BLAKE3 here — faster, but no court tooling expects it yet.

### 4c. The chain-of-custody procedure (the hash only counts inside this)

1. **Hash on receipt** — compute SHA-256 (+MD5/SHA-1) with a timestamp the instant
   the audio arrives. This is the seal.
2. **Work only on a bit-for-bit copy** — never touch the original; verify the copy's
   hash matches.
3. **Re-verify at every transfer/analysis step** — mismatch = integrity broken.
4. **Document the whole chain** — seizure, custody, transfer, analysis, disposition.
   Chain-of-custody documentation is the foundation of admissibility.
5. **If authenticity is contested**, run a SWGDE audio-authentication exam.

---

## 5. Which hash the industry actually uses (credibility slide)

| System | Hash | Use |
|---|---|---|
| Bitcoin | SHA-256 (+RIPEMD-160) | block hashing, Proof-of-Work |
| Ethereum | Keccak-256 (sponge) | addresses, tx, contract bytecode |
| Git (GitHub/GitLab) | SHA-1 → migrating to SHA-256 | commit/object IDs |
| TLS/HTTPS certificates (all public CAs) | SHA-256 | certificate signatures |
| Vercel Turborepo, IPFS/Iroh | BLAKE3 | fast content addressing / caching |
| C2PA / Content Credentials (Adobe, MS, BBC) | SHA-256/384/512 + X.509 | tamper-evident media provenance |

Two headline facts:
- **SHA-256 is the default of the internet** — every HTTPS cert is SHA-256-signed;
  the CA/Browser Forum *bans* SHA-1 (hard browser failures since 2017).
- **SHA-1 is dead for security** — the 2017 *SHAttered* attack produced a real
  collision; NIST disallowed it for signatures in 2013. That's why Git is migrating.

---

## 6. Recent advancements we could incorporate (forward-looking close)

1. **BLAKE3** — 4–10× faster than SHA-256, parallel across cores; used by
   Vercel/IPFS. Good for hashing large audio at scale. Caveat: not FIPS-approved.
2. **Post-quantum hash-based signatures — NIST FIPS 205 / SLH-DSA (finalized Aug
   2024)**, formerly SPHINCS+. Security rests *entirely on hash functions* (Merkle
   trees + WOTS+ + FORS). Matters because evidence/provenance must stay verifiable
   for *decades* — into the quantum era. Trade-off: large signatures (7–17 KB), so
   usually deployed **hybrid** (classical + PQC).
3. **SHA-3 / SHAKE** — sponge construction, resists length-extension attacks that
   plain SHA-256 is vulnerable to; SHAKE gives arbitrary-length output (used inside
   FIPS 205). A structurally-different standby to SHA-2.
4. **Merkle-tree hashing** — verify *part* of a large file/stream without re-hashing
   the whole thing (used by BLAKE3, IPFS, C2PA). Useful for long recordings.

---

## 7. How it ties back to OUR project (the synthesis)

We built a per-user audio **watermarking** service (AudioSeal / AWARE / Timbre).
For an evidence/integrity use case we add a thin **integrity + custody layer**:

- On ingest: compute SHA-256 (primary) + MD5 (compat), store
  `{hash, timestamp, source, operator}` as an immutable custody record.
- **Hash the ORIGINAL first, before watermarking** — embedding a watermark changes
  the bytes, so preserve and seal the original separately, or the watermark itself
  looks like tampering.
- Optionally sign the custody record (X.509 + SHA-256), and consider a **hybrid
  post-quantum (FIPS 205 / SLH-DSA)** signature for long-term admissibility.
- Keep the three claims separate:
  - **Hash** → file integrity since receipt.
  - **Watermark** → origin / provenance.
  - **Authentication exam** → the recording wasn't edited before receipt.

---

## 8. Anticipated questions (be ready)

- *"Why not MD5, it's faster?"* — Broken for collisions since 2004; an attacker can
  forge a different file with the same MD5. Fine as a supplementary checksum, never
  as sole legal proof.
- *"Can't someone just recompute the hash after editing?"* — Yes — that's why a hash
  alone isn't proof. It only has force *inside a documented chain of custody* where
  we recorded the hash at a trusted point in time (ideally signed/timestamped).
- *"Does the watermark replace the hash?"* — No. Different jobs: watermark survives
  re-encoding and proves origin; hash proves the exact bytes are unchanged. Court
  work wants both.
- *"Quantum computers?"* — Hash functions themselves are relatively quantum-resistant
  (Grover only halves security → SHA-256 still ~128-bit). The *signatures* are the
  weak point, which is what FIPS 205 fixes.

---

## Sources (cite these; prefer the primary ones for anything formal)

**Primary / authoritative:**
- SWGDE Best Practices for Digital Audio Authentication — https://www.swgde.org/documents/published-complete-listing/15-a-001-swgde-best-practices-for-digital-audio-authentication/
- SWGDE Best Practices for Forensic Audio v2.5 (PDF) — https://www.swgde.org/wp-content/uploads/2023/11/2022-06-09-SWGDE-Best-Practices-for-Forensic-Audio_v2.5.pdf
- SWGDE Best Practices for Digital Evidence Collection — https://www.swgde.org/documents/published-complete-listing/18-f-002-best-practices-for-digital-evidence-collection/
- NIST Computer Forensics Tool Testing (CFTT) — https://www.nist.gov/document/cftt-pres-computer-forensic-tool-testing-nist-feb-2004
- NIST FIPS 205 (SLH-DSA) full standard (PDF) — https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.205.pdf
- NIST — First 3 finalized post-quantum standards (Aug 2024) — https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards
- C2PA Technical Specification — https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- CA/Browser Forum baseline (certificate contents) — https://cabforum.org/working-groups/server/baseline-requirements/certificate-contents/
- Git hash-function transition (kernel.org) — https://www.kernel.org/pub/software/scm/git/docs/technical/hash-function-transition.html

**Supporting / explainer:**
- Acoustic fingerprint (Wikipedia) — https://en.wikipedia.org/wiki/Acoustic_fingerprint
- How Shazam works — https://www.cameronmacleod.com/blog/how-does-shazam-work
- Chromaprint / AcoustID — https://acoustid.org/chromaprint
- Digital evidence preservation standards — https://truescreen.io/articles/digital-evidence-preservation-standards/
- ISO 27037 chain-of-custody guide — https://truescreen.io/articles/digital-chain-of-custody-guide/
- FTK Imager & NIST CFTT (SHA-256 default, MD5/SHA-1/SHA-256) — https://www.examcollection.com/blog/mastering-disk-image-acquisition-in-digital-forensics-with-ftk-imager/
- DHS FTK Imager test results (PDF) — https://www.dhs.gov/sites/default/files/publications/test_results_for_ftk_imager_version_4.3.0.18_with_coverjd1gd2.pdf
- SHA-256 vs BLAKE3 — https://ssojet.com/compare-hashing-algorithms/sha-256-vs-blake3
- SHA-2 vs SHA-3 vs BLAKE3 (kerkour) — https://kerkour.com/fast-secure-hash-function-sha256-sha512-sha3-blake3
- SHA-256 vs Keccak-256 — https://financefeeds.com/cryptographic-sha256-vs-keccak256-hash-functions/
- SPHINCS+ (Wikipedia) — https://en.wikipedia.org/wiki/SPHINCS%2B

*Note: the SWGDE PDFs, NIST CFTT/FIPS 205, ISO 27037, CA/Browser Forum, and Git
kernel.org docs are the authorities. TrueScreen / ExamCollection / SSOJet are
accurate explainers — fine for understanding, but cite the primary docs in any
formal/legal write-up. Case-law admissibility (Daubert / Rule 702 applied to
hashing) was not independently verified — check for your jurisdiction if needed.*

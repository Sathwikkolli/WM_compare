"""CASP decides WHERE the watermark goes; MPEG decides HOW MUCH (its total energy).

The watermark is the same keyed, multibit, blind-detectable one as embed.py: one PN
block per 512-sample block, sign = payload bit (b mod 16), Hann overlap-add. Only the
per-bin level of each block changes:

    level(block, bin) = base_level(block, bin) + gain_db(band(bin), tile(block))

  base "mpeg" (Way 1): MPEG model-1 masking threshold + temporal envelope (= embed.py)
  base "flat" (Way 2): one flat level for 250 Hz - 8 kHz, nothing below, no envelope

Tiles are N_BANDS ERB-equal bands (250 Hz - 8 kHz, CASP's range) x TILE_S blocks.
Placement loop (place_with_casp):
  1. synthesise the watermark with the current tile gains, scale it to MPEG's energy
  2. one CASP run on host + watermark -> how noticeable it is in each tile
  3. turn noticeable tiles down and hidden tiles up, towards "equally noticeable"
  repeat N_ROUNDS[way] times. The final watermark always has exactly MPEG's energy.
Below 250 Hz CASP has no channels: Way 1 keeps MPEG's level there (gain 0 dB), Way 2
puts nothing there.
"""

import numpy as np

import casp_model as CM
from embed import FRAME_LEN, HOP, SR, block_starts, pn_block, shape_to_mask
from psychoacoustic import masking_threshold, temporal_envelope

N_BANDS = 12
BAND_LOW_HZ, BAND_HIGH_HZ = 250.0, 8000.0
TILE_S = 0.128                 # 128 ms = 8 hops of 256 samples at 16 kHz
# Rounds and step size per way, tuned on a synthetic clip: Way 1 starts close to a good
# answer (a step of 1 converges in ~4 rounds; larger steps oscillate). Way 2 starts flat
# and needs bigger steps (x3) and more rounds to get anywhere.
N_ROUNDS = {1: 4, 2: 6}
STEP = {1: 1.0, 2: 3.0}
MAX_GAIN_DB = 45.0             # tile gains stay within +-45 dB of their average
FLAT_LEVEL_DB = 40.0           # Way 2 starting level per bin (dB SPL); rescaled anyway


# --- 14. tile grid ---------------------------------------------------------------

def tile_grid(n_samples: int, fs: int = SR, n_bands: int = N_BANDS) -> dict:
    """Bands x time tiles, and how FFT bins, CASP channels and MPEG blocks map onto them."""
    e_edges = np.linspace(CM.hz_to_erb_rate(BAND_LOW_HZ), CM.hz_to_erb_rate(BAND_HIGH_HZ),
                          n_bands + 1)
    e_centres = 0.5 * (e_edges[:-1] + e_edges[1:])
    bin_hz = np.arange(FRAME_LEN // 2 + 1) * fs / FRAME_LEN
    cf = CM.erb_channels()
    tile_len = int(round(TILE_S * fs))
    n_tiles = int(np.ceil(n_samples / tile_len))
    starts = np.array(list(block_starts(n_samples)))
    return dict(
        n_bands=n_bands, n_tiles=n_tiles, tile_len=tile_len,
        e_centres=e_centres,
        bin_erb=CM.hz_to_erb_rate(bin_hz),
        bin_in_range=(bin_hz >= BAND_LOW_HZ),          # bins CASP can judge
        chan_band=np.clip(np.searchsorted(e_edges, CM.hz_to_erb_rate(cf), side="right") - 1,
                          0, n_bands - 1),
        # fractional tile position of each MPEG block's centre (for smooth interpolation)
        block_tile=(starts + FRAME_LEN / 2) / tile_len - 0.5,
        tile_centres=np.arange(n_tiles, dtype=float),
    )


def _gain_db_per_bin(gain_db: np.ndarray, grid: dict) -> np.ndarray:
    """Tile gains (bands, tiles) -> per-block, per-bin gains (blocks, bins), smoothly
    interpolated in time (between tile centres) and in frequency (between band centres,
    on the ERB axis), so no hard steps appear at tile edges. Bins below 250 Hz get 0 dB."""
    per_block = np.stack([np.interp(grid["block_tile"], grid["tile_centres"], gain_db[k])
                          for k in range(grid["n_bands"])], axis=1)          # (blocks, bands)
    out = np.stack([np.interp(grid["bin_erb"], grid["e_centres"], row) for row in per_block])
    out[:, ~grid["bin_in_range"]] = 0.0
    return out


# --- 16. base levels: what the watermark looks like before CASP moves it -----------

def base_levels(host: np.ndarray, base: str, fs: int = SR) -> tuple:
    """Per-block target level (blocks, bins) in dB SPL and per-block temporal weight
    (blocks, samples). "mpeg" reproduces embed.embed_watermark exactly; "flat" is one
    level everywhere in 250 Hz - 8 kHz (a starting point CASP then reshapes)."""
    host = np.asarray(host, dtype=float)
    starts = list(block_starts(len(host)))
    if base == "mpeg":
        levels = np.stack([masking_threshold(host[s:s + FRAME_LEN], fs) for s in starts])
        env = temporal_envelope(host, fs)
        weights = np.stack([env[s:s + FRAME_LEN] / max(env[s:s + FRAME_LEN].max(), 1e-9)
                            for s in starts])
    elif base == "flat":
        bin_hz = np.arange(FRAME_LEN // 2 + 1) * fs / FRAME_LEN
        row = np.where(bin_hz >= BAND_LOW_HZ, FLAT_LEVEL_DB, -200.0)
        levels = np.tile(row, (len(starts), 1))
        weights = np.ones((len(starts), FRAME_LEN))
    else:
        raise ValueError(base)
    return levels, weights


def synthesise(n_samples: int, bits, key: int, levels: np.ndarray, weights: np.ndarray,
               gain_db_bins=None) -> np.ndarray:
    """The watermark alone: sum over blocks of sign * PN shaped to (level + gain) *
    Hann * temporal weight. Same PN, signs and overlap-add as embed.embed_watermark."""
    signs = 2.0 * np.asarray(bits, dtype=float) - 1.0
    synth = np.hanning(FRAME_LEN + 1)[:-1]
    wm = np.zeros(n_samples)
    for b, s in enumerate(block_starts(n_samples)):
        lvl = levels[b] if gain_db_bins is None else levels[b] + gain_db_bins[b]
        w = shape_to_mask(pn_block(key, b), lvl)
        wm[s:s + FRAME_LEN] += signs[b % len(signs)] * w * synth * weights[b]
    return wm


# --- 15. audibility map ----------------------------------------------------------

def audibility_map(ir_host: dict, ir_marked: dict, grid: dict) -> np.ndarray:
    """How noticeable the watermark is in each tile, in JND units (bands, tiles).

    One CASP run gives the change of the internal picture per channel and time; summing
    the squared change over modulation filters, over the channels of a band and over the
    IR samples of a tile gives each tile's share: total d'^2 = sum of the tiles' d'^2.
    """
    d2 = np.sum((ir_marked["ir"] - ir_host["ir"]) ** 2, axis=1)          # (channels, time_ir)
    fs_ir = ir_host["fs_ir"]
    tile_ir = grid["tile_len"] * fs_ir / SR
    t_tile = np.minimum((np.arange(d2.shape[1]) / tile_ir).astype(int), grid["n_tiles"] - 1)
    out = np.zeros((grid["n_bands"], grid["n_tiles"]))
    np.add.at(out, (grid["chan_band"][:, None], t_tile[None, :]), d2)
    return np.sqrt(out / fs_ir) / CM.calibration_distance(SR)


# --- 17/18. gains and energy --------------------------------------------------------

def match_energy(wm: np.ndarray, target_energy: float) -> tuple:
    """Scale the watermark so sum(wm^2) == target_energy. Returns (wm, scale)."""
    e = float(np.sum(wm ** 2))
    s = np.sqrt(target_energy / e) if e > 0 else 0.0
    return wm * s, s


def _recentre(gain_db: np.ndarray) -> np.ndarray:
    """Keep gains within +-MAX_GAIN_DB of their mean (the overall level is set by energy)."""
    m = gain_db.mean()
    return np.clip(gain_db, m - MAX_GAIN_DB, m + MAX_GAIN_DB) - m


# --- 19. the placement loop ---------------------------------------------------------

def place_with_casp(host: np.ndarray, bits, key: int, base: str, target_energy: float,
                    n_rounds: int, step: float = 1.0, fs: int = SR, ir_host=None):
    """Move the watermark's energy between tiles until every tile is about equally
    noticeable to CASP, at a fixed total energy. Returns (gain_db, info)."""
    host = np.asarray(host, dtype=float)
    grid = tile_grid(len(host), fs)
    levels, weights = base_levels(host, base, fs)
    ir_host = ir_host or CM.internal_representation(host, fs)
    gain_db = np.zeros((grid["n_bands"], grid["n_tiles"]))
    history = []
    for r in range(n_rounds + 1):                       # last pass only measures
        wm = synthesise(len(host), bits, key, levels, weights, _gain_db_per_bin(gain_db, grid))
        wm, _ = match_energy(wm, target_energy)
        aud = audibility_map(ir_host, CM.internal_representation(host + wm, fs), grid)
        history.append(dict(round=r, d_total=float(np.sqrt(np.sum(aud ** 2))),
                            d_max_tile=float(aud.max()), spread_db=float(np.ptp(gain_db))))
        if r == n_rounds:
            break
        # Audibility grows ~linearly with the tile's amplitude, so scaling a tile by
        # (reference / its audibility) brings it towards the reference: equal audibility.
        live = aud > 1e-9 * max(aud.max(), 1e-30)
        ref = np.exp(np.mean(np.log(aud[live]))) if live.any() else 1.0
        change = np.zeros_like(gain_db)
        change[live] = 20 * np.log10(ref / aud[live])
        gain_db = _recentre(gain_db + step * change)
    return gain_db, dict(grid=grid, history=history, ir_host=ir_host)


# --- 20. the fairness control -----------------------------------------------------

def shuffle_gains(gain_db: np.ndarray, seed: int) -> np.ndarray:
    """Same tile gains, handed out to random tiles: keeps how uneven the map is, destroys
    where CASP put the energy."""
    rng = np.random.default_rng([seed, 4242])
    return rng.permutation(gain_db.ravel()).reshape(gain_db.shape)


# --- 21. the embedder -------------------------------------------------------------

def mpeg_energy(host: np.ndarray, bits, key: int, fs: int = SR) -> float:
    """Energy of the plain MPEG watermark for this clip, key and payload: the budget."""
    levels, weights = base_levels(host, "mpeg", fs)
    return float(np.sum(synthesise(len(host), bits, key, levels, weights) ** 2))


def embed_casp(host: np.ndarray, bits, key: int, way: int = 1, shuffle: bool = False,
               gain_db=None, n_rounds=None, fs: int = SR):
    """Embed the 16-bit payload with CASP placement at MPEG's energy.

    way 1: start from the MPEG watermark, CASP corrects it. way 2: start flat, CASP
    decides everything. shuffle: use the same tile gains in random tiles (control).
    gain_db: reuse tile gains already computed for this clip (skips the CASP rounds).
    Returns (marked, info) with info["gain_db"] the (unshuffled) tile gains.
    """
    host = np.asarray(host, dtype=float)
    base = {1: "mpeg", 2: "flat"}[way]
    target = mpeg_energy(host, bits, key, fs)
    info = {}
    if gain_db is None:
        rounds = N_ROUNDS[way] if n_rounds is None else n_rounds
        gain_db, info = place_with_casp(host, bits, key, base, target, rounds, STEP[way], fs)
    used = shuffle_gains(gain_db, key) if shuffle else gain_db
    grid = tile_grid(len(host), fs)
    levels, weights = base_levels(host, base, fs)
    wm = synthesise(len(host), bits, key, levels, weights, _gain_db_per_bin(used, grid))
    wm, _ = match_energy(wm, target)
    info.update(gain_db=gain_db, target_energy=target, way=way, shuffle=shuffle)
    return (host + wm).astype(np.float32), info

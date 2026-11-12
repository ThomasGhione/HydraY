// Reference reader (pure std, no GPU) for the quantised.bin written by
// trainer_deep.rs: it is the ORACLE. The C++ must agree with this file, not the
// other way round.
//
// Net: (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 [-> L2] -> 1, with OB
// material-count output buckets on every layer after l0. QA=255 QB=64 SCALE=400.
// OB is 16 or 8 and L2 is 0 (no second hidden layer) or L2_SIZE, all recognised by
// the file size; the inputs are those of sanity.rs.
//
// LAYOUT (little-endian, save_format order, padded to 64 B with "bullet"):
//
//   offset              field                  type   scale
//   0                   l0w [4*768][1024]      i16    QA
//   6,291,456           l0b [1024]             i16    QA
//   6,293,504           l1w [OB][16][1024]     i8     QB
//   + OB*16384          l1b [OB][16]           f32    real
//   + OB*64             l2w [OB][1][16]        f32    real
//   + OB*64             l2b [OB]               f32    real
//   OB = 16: payload and file 6,557,760; OB = 8: payload 6,425,632, file 6,425,664
//
// WITH THE SECOND HIDDEN LAYER (L2 = L2_SIZE = 32), the tail after l1b is instead:
//
//   l2w [OB][L2][16]       f32    real   (transposed)
//   l2b [OB][L2]           f32    real
//   l3w [OB][1][L2]        f32    real   (transposed)
//   l3b [OB]               f32    real
//   OB = 8: payload 6,443,552, file 6,443,584; OB = 16: payload and file 6,593,600
//
// The whole tail after l1w stays float, as l2 always has.
//
// ARITHMETIC, the delicate part:
//
//   c[i]  = clamp(acc[i], 0, QA)                    QA scale, fits in u8
//   p[j]  = c[j] * c[j + 512] / QA                  pairwise, back in u8
//   hl1   = [p_stm (512), p_ntm (512)]              1024 values, QA scale
//   z[o]  = (SUM hl1[i]*l1w[o][i]) / (QA*QB) + l1b[o]     real
//   a[o]  = clamp(z[o], 0, 1)^2                     SCReLU in float
//   y     = SUM a[o]*l2w[o] + l2b                   linear (L2 = 0)
//   or, with L2:
//   a2[k] = clamp(SUM a[o]*l2w[k][o] + l2b[k], 0, 1)^2   SCReLU in float
//   y     = SUM a2[k]*l3w[k] + l3b                  linear
//   cp    = y * SCALE
//
// Usage: cargo run -r --bin sanity_deep -- <quantised.bin> [fen... | --init-check]

const HIDDEN: usize = 1024;
const L1_SIZE: usize = 16;
// Width of the optional second hidden layer; a file has it or not (L2 = 0).
pub const L2_SIZE: usize = 32;
const L2_WIDTHS: [usize; 2] = [0, L2_SIZE];
const INPUT_BUCKETS: usize = 4;
// The output-bucket counts a file may have, as in network_deep.hpp.
const OUTPUT_BUCKET_COUNTS: [usize; 2] = [16, 8];
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: f32 = 400.0;

// Keep in sync with trainer_deep.rs BUCKET_LAYOUT and nnue/network.hpp.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];

fn king_bucket(ksq: usize) -> usize {
    const FILE_FOLD: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];
    BUCKET_LAYOUT[(ksq / 8) * 4 + FILE_FOLD[ksq % 8]]
}

pub struct Network {
    pub ob: usize,      // output buckets
    pub l2: usize,      // width of the second hidden layer, 0 if none
    pub l0w: Vec<i16>,  // [4*768 * HIDDEN]
    pub l0b: Vec<i16>,  // [HIDDEN]
    pub l1w: Vec<i8>,   // [OB * L1 * HIDDEN]
    pub l1b: Vec<f32>,  // [OB * L1]
    pub l2w: Vec<f32>,  // [OB * L1] if l2 == 0, else [OB * L2 * L1]
    pub l2b: Vec<f32>,  // [OB] if l2 == 0, else [OB * L2]
    pub l3w: Vec<f32>,  // [OB * L2], empty if l2 == 0
    pub l3b: Vec<f32>,  // [OB], empty if l2 == 0
}

pub const L0W: usize = INPUT_BUCKETS * 768 * HIDDEN;

pub const fn payload(ob: usize, l2: usize) -> usize {
    let head = ob * L1_SIZE * HIDDEN + ob * L1_SIZE * 4;
    let tail = if l2 == 0 { ob * L1_SIZE * 4 + ob * 4 } else { ob * l2 * L1_SIZE * 4 + ob * l2 * 4 + ob * l2 * 4 + ob * 4 };
    L0W * 2 + HIDDEN * 2 + head + tail
}

pub const fn file_bytes(ob: usize, l2: usize) -> usize {
    payload(ob, l2).div_ceil(64) * 64
}

pub fn load(path: &str) -> Network {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let layouts: Vec<(usize, usize)> =
        OUTPUT_BUCKET_COUNTS.into_iter().flat_map(|ob| L2_WIDTHS.into_iter().map(move |l2| (ob, l2))).collect();
    let (ob, l2) = layouts.iter().copied().find(|&(ob, l2)| bytes.len() == file_bytes(ob, l2)).unwrap_or_else(|| {
        let known: Vec<String> =
            layouts.iter().map(|&(ob, l2)| format!("{} ({ob} buckets, l2 {l2})", file_bytes(ob, l2))).collect();
        panic!("size mismatch: got {} bytes, expected one of {}", bytes.len(), known.join(", "))
    });
    let end = payload(ob, l2);
    // bullet fills the tail with "bullet" repeated: anything else means the
    // layout has drifted.
    assert!(
        bytes[end..].iter().zip(b"bullet".iter().cycle()).all(|(a, b)| a == b),
        "unexpected padding tail - layout drift?"
    );

    let mut off = 0usize;
    let i16s = |n: usize, off: &mut usize| -> Vec<i16> {
        let v = bytes[*off..*off + n * 2]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        *off += n * 2;
        v
    };
    let l0w = i16s(L0W, &mut off);
    let l0b = i16s(HIDDEN, &mut off);

    let l1w_len = ob * L1_SIZE * HIDDEN;
    let l1w: Vec<i8> = bytes[off..off + l1w_len].iter().map(|&b| b as i8).collect();
    off += l1w_len;

    let f32s = |n: usize, off: &mut usize| -> Vec<f32> {
        let v = bytes[*off..*off + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        *off += n * 4;
        v
    };
    let l1b = f32s(ob * L1_SIZE, &mut off);
    let (l2w, l2b, l3w, l3b) = if l2 == 0 {
        (f32s(ob * L1_SIZE, &mut off), f32s(ob, &mut off), Vec::new(), Vec::new())
    } else {
        let l2w = f32s(ob * l2 * L1_SIZE, &mut off);
        let l2b = f32s(ob * l2, &mut off);
        let l3w = f32s(ob * l2, &mut off);
        let l3b = f32s(ob, &mut off);
        (l2w, l2b, l3w, l3b)
    };
    assert_eq!(off, end, "parsing did not consume exactly the payload");

    Network { ob, l2, l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b }
}

fn screlu_f32(x: f32) -> f32 {
    let y = x.clamp(0.0, 1.0);
    y * y
}

/// The active features of both perspectives and the output bucket (of `ob`), from
/// the FEN. Identical to sanity.rs: the part the two architectures share.
pub fn features(ob: usize, fen: &str) -> (Vec<usize>, Vec<usize>, usize) {
    let mut fields = fen.split_whitespace();
    let board = fields.next().expect("empty FEN");
    let black_to_move = fields.next() == Some("b");

    let mut pieces: Vec<(usize, bool, usize)> = Vec::with_capacity(32);
    let (mut wk, mut bk) = (usize::MAX, usize::MAX);
    for (rank_from_top, row) in board.split('/').enumerate() {
        let rank = 7 - rank_from_top;
        let mut file = 0usize;
        for c in row.chars() {
            if let Some(skip) = c.to_digit(10) {
                file += skip as usize;
                continue;
            }
            let piece_type = match c.to_ascii_lowercase() {
                'p' => 0, 'n' => 1, 'b' => 2, 'r' => 3, 'q' => 4, 'k' => 5,
                other => panic!("bad FEN piece '{other}'"),
            };
            let is_black_piece = c.is_ascii_lowercase();
            let sq = rank * 8 + file;
            if piece_type == 5 {
                if is_black_piece { bk = sq } else { wk = sq }
            }
            pieces.push((piece_type, is_black_piece, sq));
            file += 1;
        }
    }
    assert!(wk != usize::MAX && bk != usize::MAX, "FEN without both kings");

    let params = |from_black_view: bool| {
        let own = if from_black_view { bk ^ 56 } else { wk };
        let flip = if own % 8 > 3 { 7 } else { 0 };
        (768 * king_bucket(own), flip)
    };
    let (us_base, us_flip) = params(black_to_move);
    let (them_base, them_flip) = params(!black_to_move);

    let mut us = Vec::with_capacity(pieces.len());
    let mut them = Vec::with_capacity(pieces.len());
    for &(piece_type, is_black_piece, sq) in &pieces {
        for (list, from_black_view, base, flip) in [
            (&mut us, black_to_move, us_base, us_flip),
            (&mut them, !black_to_move, them_base, them_flip),
        ] {
            let is_opp = is_black_piece != from_black_view;
            let sq_view = if from_black_view { sq ^ 56 } else { sq };
            let feat768 = usize::from(is_opp) * 384 + piece_type * 64 + sq_view;
            list.push(base + (feat768 ^ flip));
        }
    }

    // bullet MaterialCount<ob>.
    let out_bucket = (pieces.len() - 2) / 32usize.div_ceil(ob);
    (us, them, out_bucket)
}

/// The two perspectives' accumulators and the output bucket (of `ob`), from the FEN.
pub fn accumulate(net_l0w: &[i16], net_l0b: &[i16], ob: usize, fen: &str) -> (Vec<i16>, Vec<i16>, usize) {
    let (us, them, out_bucket) = features(ob, fen);
    let sum = |list: &[usize]| {
        let mut acc = net_l0b.to_vec();
        for &f in list {
            for (a, w) in acc.iter_mut().zip(&net_l0w[f * HIDDEN..(f + 1) * HIDDEN]) {
                *a += *w;
            }
        }
        acc
    };
    (sum(&us), sum(&them), out_bucket)
}

/// CReLU + pairwise product: 1024 i16 values -> 512 values in [0, QA].
fn pairwise(acc: &[i16], out: &mut [i32]) {
    let half = HIDDEN / 2;
    for j in 0..half {
        let a = i32::from(acc[j]).clamp(0, QA);
        let b = i32::from(acc[j + half]).clamp(0, QA);
        out[j] = a * b / QA;
    }
}

pub fn evaluate(net: &Network, fen: &str) -> i32 {
    let (acc_us, acc_them, ob) = accumulate(&net.l0w, &net.l0b, net.ob, fen);

    let half = HIDDEN / 2;
    let mut hl1 = vec![0i32; HIDDEN];
    pairwise(&acc_us, &mut hl1[..half]);
    pairwise(&acc_them, &mut hl1[half..]);

    // l1: [L1_SIZE][HIDDEN] per bucket, already transposed at save time.
    let w1 = &net.l1w[ob * L1_SIZE * HIDDEN..(ob + 1) * L1_SIZE * HIDDEN];
    let b1 = &net.l1b[ob * L1_SIZE..(ob + 1) * L1_SIZE];
    let mut a1 = [0f32; L1_SIZE];
    for o in 0..L1_SIZE {
        let row = &w1[o * HIDDEN..(o + 1) * HIDDEN];
        let mut sum = 0i32;
        for (h, w) in hl1.iter().zip(row) {
            sum += h * i32::from(*w);
        }
        a1[o] = screlu_f32(sum as f32 / (QA * QB) as f32 + b1[o]);
    }

    (head(net, ob, &a1) * SCALE).round() as i32
}

/// Everything after l1, in float: l2 -> 1, or l2 (SCReLU) -> l3 -> 1.
fn head(net: &Network, ob: usize, a1: &[f32; L1_SIZE]) -> f32 {
    if net.l2 == 0 {
        let w2 = &net.l2w[ob * L1_SIZE..(ob + 1) * L1_SIZE];
        return net.l2b[ob] + a1.iter().zip(w2).map(|(a, w)| a * w).sum::<f32>();
    }
    let l2 = net.l2;
    let mut y = net.l3b[ob];
    for k in 0..l2 {
        let row = &net.l2w[(ob * l2 + k) * L1_SIZE..(ob * l2 + k + 1) * L1_SIZE];
        let z = net.l2b[ob * l2 + k] + a1.iter().zip(row).map(|(a, w)| a * w).sum::<f32>();
        y += screlu_f32(z) * net.l3w[ob * l2 + k];
    }
    y
}

// ---------------------------------------------------------------------------
// The same net in bullet's weight layout, to start a training run from it
// (trainer_deep.rs, HYDRAY_INIT_NET). bullet holds a matrix column-major as
// (outputs x inputs): l0w keeps its saved order, l1w, l2w and l3w are the
// transposes of their saved form (their SavedFormat transposes them).
//
// A net without the second hidden layer can start a run that has one: l0 and l1
// come from the net, l2 and l3 are new (fresh_head), so that run begins from the
// net's feature transformer and first layer but not from its evaluation.

pub struct BulletInit {
    pub ob: usize,
    pub l0w: Vec<f32>, // [4*768][HIDDEN]; l0w + l0f is the saved weight
    pub l0f: Vec<f32>, // [768][HIDDEN], the factoriser
    pub l0b: Vec<f32>, // [HIDDEN]
    pub l1w: Vec<f32>, // [HIDDEN][OB*L1]
    pub l1b: Vec<f32>, // [OB*L1]
    pub l2: usize,     // width of the second hidden layer, 0 if none
    pub l2w: Vec<f32>, // [L1][OB] if l2 == 0, else [L1][OB*L2]
    pub l2b: Vec<f32>, // [OB] if l2 == 0, else [OB*L2]
    pub l3w: Vec<f32>, // [L2][OB], empty if l2 == 0
    pub l3b: Vec<f32>, // [OB], empty if l2 == 0
    pub clipped: usize, // l0 values past the trainer's ±0.99 clip
    pub revived: usize,
    pub fresh_head: bool, // l2 and l3 newly initialised (the net had no second layer)
}

/// Dead l1 neurons per output bucket: all-zero rows. A neuron that stops firing
/// gets no gradient, and AdamW's decay then zeroes it exactly.
pub fn dead_rows(net: &Network) -> Vec<usize> {
    (0..net.ob)
        .map(|b| {
            (0..L1_SIZE)
                .filter(|n| {
                    let o = b * L1_SIZE + n;
                    net.l1w[o * HIDDEN..(o + 1) * HIDDEN].iter().all(|&w| w == 0)
                })
                .count()
        })
        .collect()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn gaussian(state: &mut u64) -> f32 {
    let u1 = ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

/// With `revive_bias`, every dead l1 neuron gets a fresh row (bullet's l1 init),
/// that bias and a zero weight into the next layer: it fires again, and no
/// evaluation changes.
///
/// `l2` is the width of the run's second hidden layer. When the net has none and
/// `l2` > 0, l2 and l3 are initialised here (fixed seed, He-normal weights, l2's
/// bias `head_bias` so that its neurons start firing, l3's bias 0): computed here
/// rather than by bullet so that the start check can still compare the two forwards.
pub fn to_bullet(net: &Network, revive_bias: Option<f32>, l2: usize, head_bias: f32) -> BulletInit {
    assert!(net.l2 == l2 || net.l2 == 0, "the net has a second layer of {}, the run wants {l2}", net.l2);
    let (qa, qb) = (QA as f32, QB as f32);
    let folded: Vec<f32> = net.l0w.iter().map(|&w| f32::from(w) / qa).collect();
    // The factoriser takes the midrange of each feature's four bucket weights,
    // moved into the range that keeps it and every bucket part inside the clip.
    let mut l0f = vec![0f32; 768 * HIDDEN];
    for (i, f) in l0f.iter_mut().enumerate() {
        let (lo, hi) = (0..INPUT_BUCKETS)
            .map(|b| folded[b * 768 * HIDDEN + i])
            .fold((f32::MAX, f32::MIN), |(lo, hi), w| (lo.min(w), hi.max(w)));
        let (min_f, max_f) = ((hi - 0.99).max(-0.99), (lo + 0.99).min(0.99));
        let mid = (lo + hi) / 2.0;
        *f = if min_f <= max_f { mid.clamp(min_f, max_f) } else { mid };
    }
    let l0w: Vec<f32> = folded.iter().enumerate().map(|(i, w)| w - l0f[i % (768 * HIDDEN)]).collect();
    let clipped = l0w.iter().chain(&l0f).filter(|w| w.abs() > 0.99).count();

    let n1 = net.ob * L1_SIZE;
    let mut l1w = vec![0f32; HIDDEN * n1];
    for o in 0..n1 {
        for i in 0..HIDDEN {
            l1w[i * n1 + o] = f32::from(net.l1w[o * HIDDEN + i]) / qb;
        }
    }
    let mut l1b = net.l1b.clone();
    let ob = net.ob;
    let fresh_head = l2 > 0 && net.l2 == 0;
    let (mut l2w, l2b, l3w, l3b) = if l2 == 0 {
        let mut w = vec![0f32; n1];
        for b in 0..ob {
            for n in 0..L1_SIZE {
                w[n * ob + b] = net.l2w[b * L1_SIZE + n];
            }
        }
        (w, net.l2b.clone(), Vec::new(), Vec::new())
    } else if !fresh_head {
        // saved [OB][L2][L1] -> bullet (OB*L2 x L1) column-major: [L1][OB*L2]
        let n2 = ob * l2;
        let mut w2 = vec![0f32; L1_SIZE * n2];
        for o in 0..n2 {
            for n in 0..L1_SIZE {
                w2[n * n2 + o] = net.l2w[o * L1_SIZE + n];
            }
        }
        // saved [OB][1][L2] -> bullet (OB x L2): [L2][OB]
        let mut w3 = vec![0f32; l2 * ob];
        for b in 0..ob {
            for k in 0..l2 {
                w3[k * ob + b] = net.l3w[b * l2 + k];
            }
        }
        (w2, net.l2b.clone(), w3, net.l3b.clone())
    } else {
        let mut rng = 0x5eed_u64; // fixed seed: two arms start from the same head
        let n2 = ob * l2;
        let s2 = (2.0 / L1_SIZE as f32).sqrt();
        let s3 = (2.0 / l2 as f32).sqrt();
        let w2 = (0..L1_SIZE * n2).map(|_| s2 * gaussian(&mut rng)).collect();
        let w3 = (0..l2 * ob).map(|_| s3 * gaussian(&mut rng)).collect();
        (w2, vec![head_bias; n2], w3, vec![0f32; ob])
    };

    let mut revived = 0;
    if let Some(bias) = revive_bias {
        let stdev = (2.0 / HIDDEN as f32).sqrt();
        let mut rng = 0u64; // fixed seed: two arms of an A/B revive identically
        for o in 0..n1 {
            if net.l1w[o * HIDDEN..(o + 1) * HIDDEN].iter().all(|&w| w == 0) {
                for i in 0..HIDDEN {
                    l1w[i * n1 + o] = stdev * gaussian(&mut rng);
                }
                l1b[o] = bias;
                let (b, n) = (o / L1_SIZE, o % L1_SIZE);
                if l2 == 0 {
                    l2w[n * ob + b] = 0.0;
                } else if !fresh_head {
                    // its weight into every l2 neuron of its bucket
                    for k in 0..l2 {
                        l2w[n * (ob * l2) + b * l2 + k] = 0.0;
                    }
                }
                revived += 1;
            }
        }
    }

    let l0b = net.l0b.iter().map(|&w| f32::from(w) / qa).collect();
    BulletInit { ob, l0w, l0f, l0b, l1w, l1b, l2, l2w, l2b, l3w, l3b, clipped, revived, fresh_head }
}

/// Under QAT, added to the pairwise product before it is truncated. The exact
/// product c*c'*QA is either an integer (the engine keeps it whole) or at least
/// 1/QA away from one, while float error moves it by about 3e-5: a saturated input
/// (c = 1) makes it an integer often, and without the nudge truncation would land
/// one step low. Half a step of the gap, 0.5/QA in units of QA, sits safely in it.
pub const PAIRWISE_NUDGE: f32 = 0.5 / (QA * QA) as f32;

// bullet's faux_quantise: x on the grid of step 1/q, rounded or truncated.
fn faux(x: f32, q: i32, round: bool) -> f32 {
    let s = x * q as f32;
    (if round { s.round() } else { s.trunc() }) / q as f32
}

/// Float forward over the bullet layout: bullet's graph, without the rounding of
/// the quantised net. In centipawns, stm-relative. With `qat`, the graph of
/// trainer_deep.rs under HYDRAY_QAT: l0 (factoriser folded) and l0b rounded to
/// 1/QA, the pairwise product truncated to 1/QA, l1w rounded to 1/QB, which is
/// the quantised net's arithmetic in float.
pub fn evaluate_bullet(b: &BulletInit, fen: &str, qat: bool) -> f32 {
    let (us, them, ob) = features(b.ob, fen);
    let half = HIDDEN / 2;
    let q = |x: f32, scale: i32, round: bool| if qat { faux(x, scale, round) } else { x };
    let ft = |list: &[usize]| -> Vec<f32> {
        let mut acc: Vec<f32> = b.l0b.iter().map(|&w| q(w, QA, true)).collect();
        for &f in list {
            let (w, fac) = (&b.l0w[f * HIDDEN..(f + 1) * HIDDEN], &b.l0f[(f % 768) * HIDDEN..(f % 768 + 1) * HIDDEN]);
            for h in 0..HIDDEN {
                acc[h] += q(w[h] + fac[h], QA, true);
            }
        }
        (0..half)
            .map(|j| {
                let p = acc[j].clamp(0.0, 1.0) * acc[j + half].clamp(0.0, 1.0);
                if qat { faux(p + PAIRWISE_NUDGE, QA, false) } else { p }
            })
            .collect()
    };
    let hl1: Vec<f32> = ft(&us).into_iter().chain(ft(&them)).collect();

    let n1 = b.ob * L1_SIZE;
    let a1: Vec<f32> = (0..L1_SIZE)
        .map(|n| {
            let o = ob * L1_SIZE + n;
            screlu_f32(b.l1b[o] + hl1.iter().enumerate().map(|(i, h)| h * q(b.l1w[i * n1 + o], QB, true)).sum::<f32>())
        })
        .collect();
    let y = if b.l2 == 0 {
        b.l2b[ob] + (0..L1_SIZE).map(|n| a1[n] * b.l2w[n * b.ob + ob]).sum::<f32>()
    } else {
        let n2 = b.ob * b.l2;
        b.l3b[ob]
            + (0..b.l2)
                .map(|k| {
                    let o = ob * b.l2 + k;
                    let z = b.l2b[o] + (0..L1_SIZE).map(|n| a1[n] * b.l2w[n * n2 + o]).sum::<f32>();
                    screlu_f32(z) * b.l3w[k * b.ob + ob]
                })
                .sum::<f32>()
    };
    y * SCALE
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("quantised.bin");
    let net = load(path);
    println!("{path}: layout OK ({} output buckets, second hidden layer {}, {} B payload)", net.ob, net.l2, payload(net.ob, net.l2));
    println!("dead l1 neurons (all-zero rows) per output bucket: {:?}", dead_rows(&net));

    let default_fens = [
        ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("wN e4 extra (w)", "rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("bN e5 extra (b) [must equal previous]", "rnbqkbnr/pppppppp/8/4n3/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
        ("middlegame ~24 pieces (w)", "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"),
        ("KRPvKR (w)", "8/8/4k3/8/2r5/2K5/4P3/3R4 w - - 0 1"),
        ("KQvK (w)", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"),
        ("both castled short (w)", "r4rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 1"),
        ("white long castle vs e8 king (w)", "r3kb1r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPPBQPPP/2KR3R w kq - 0 1"),
        ("kings on 2nd rank (w)", "8/1k3ppp/1p6/p1p5/P1P5/1P4P1/1K3P1P/8 w - - 0 1"),
        ("active kings endgame (w)", "8/8/3k4/8/2p5/2P1K3/8/8 w - - 0 1"),
    ];

    if args.get(2).map(String::as_str) == Some("--init-check") {
        // What trainer_deep.rs starts from with HYDRAY_INIT_NET: the float forward
        // must match the quantised one up to the rounding of the quantised net, and
        // the QAT forward must match it exactly but for the l0 values past the clip.
        for revive in [None, Some(0.1)] {
            let b = to_bullet(&net, revive, net.l2, 0.1);
            let worst = |qat| {
                default_fens
                    .iter()
                    .map(|(_, fen)| (evaluate_bullet(&b, fen, qat) - evaluate(&net, fen) as f32).abs())
                    .fold(0.0, f32::max)
            };
            println!(
                "bullet layout, {} neurons revived: {} l0 values past the clip, worst |float - quantised| {:.2} cp, \
                 |qat - quantised| {:.2} cp",
                b.revived, b.clipped, worst(false), worst(true)
            );
        }
        return;
    }

    if args.len() > 2 {
        for fen in &args[2..] {
            println!("{fen}: {} cp (stm)", evaluate(&net, fen));
        }
    } else {
        for (label, fen) in default_fens {
            println!("{label}: {} cp (stm)", evaluate(&net, fen));
        }
    }
}

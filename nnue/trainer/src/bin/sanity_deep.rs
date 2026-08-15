// Lettore di riferimento (pure-std, niente GPU) per il quantised.bin prodotto
// da trainer_deep.rs — la rete con layer intermedi.
//
// Come sanity.rs sta alla rete a un layer, questo sta alla rete profonda: e'
// l'ORACOLO. Il C++ deve concordare con questo file, non viceversa, e va
// scritto dopo averlo letto.
//
// Rete: (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 1,
// 8 output bucket applicati sia a l1 sia a l2. QA=255 QB=64 SCALE=400.
//
// Input identici alla rete a un layer (4 king bucket specchiati, 8 output
// bucket su popcount): quella parte e' copiata da sanity.rs apposta, cosi' un
// eventuale disaccordo fra i due lettori isola la sola parte nuova.
//
// LAYOUT (little-endian, nell'ordine di save_format, pad a 64 B con "bullet"):
//
//   offset          campo                      tipo   scala
//   0               l0w [4*768][1024]          i16    QA
//   6.291.456       l0b [1024]                 i16    QA
//   6.293.504       l1w [8][16][1024]          i8     QB
//   6.424.576       l1b [8][16]                f32    reale
//   6.425.088       l2w [8][1][16]             f32    reale
//   6.425.600       l2b [8]                    f32    reale
//   6.425.632       fine payload (pad a 6.425.664)
//
// ARITMETICA, che e' il punto delicato:
//
//   acc[i]              i16, scala QA
//   c[i]  = clamp(acc[i], 0, QA)                    scala QA, sta in u8
//   p[j]  = c[j] * c[j + 512] / QA                  scala QA, sta in u8
//                       ^ e' il pairwise: da 1024 valori per prospettiva a 512,
//                         e la divisione per QA e' cio' che lo rimette in u8.
//   hl1   = [p_stm (512), p_ntm (512)]              1024 valori, scala QA
//   z[o]  = (SUM hl1[i]*l1w[o][i]) / (QA*QB) + l1b[o]     reale
//   a[o]  = clamp(z[o], 0, 1)^2                     SCReLU in float
//   y     = SUM a[o]*l2w[o] + l2b        lineare
//   cp    = y * SCALE
//
// Uso: cargo run -r --bin sanity_deep -- <quantised.bin> [fen]...

const HIDDEN: usize = 1024;
const L1_SIZE: usize = 16;
const INPUT_BUCKETS: usize = 4;
const OUTPUT_BUCKETS: usize = 8;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: f32 = 400.0;

// Keep in sync with trainer_deep.rs BUCKET_LAYOUT e nnue/network.hpp.
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
    pub l0w: Vec<i16>,  // [4*768 * HIDDEN]
    pub l0b: Vec<i16>,  // [HIDDEN]
    pub l1w: Vec<i8>,   // [OB * L1 * HIDDEN]
    pub l1b: Vec<f32>,  // [OB * L1]
    pub l2w: Vec<f32>,  // [OB * L1]
    pub l2b: Vec<f32>,  // [OB]
}

pub const L0W: usize = INPUT_BUCKETS * 768 * HIDDEN;
pub const L1W: usize = OUTPUT_BUCKETS * L1_SIZE * HIDDEN;
pub const PAYLOAD: usize = L0W * 2
    + HIDDEN * 2
    + L1W
    + OUTPUT_BUCKETS * L1_SIZE * 4
    + OUTPUT_BUCKETS * L1_SIZE * 4
    + OUTPUT_BUCKETS * 4;

pub fn load(path: &str) -> Network {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert!(
        bytes.len() >= PAYLOAD && bytes.len() % 64 == 0,
        "size mismatch: got {} bytes, expected {PAYLOAD} (+pad to 64)",
        bytes.len()
    );
    // bullet riempie la coda con "bullet" ripetuto: qualsiasi altra cosa
    // significa che il layout e' andato alla deriva.
    assert!(
        bytes[PAYLOAD..].iter().zip(b"bullet".iter().cycle()).all(|(a, b)| a == b),
        "unexpected padding tail — layout drift?"
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

    let l1w: Vec<i8> = bytes[off..off + L1W].iter().map(|&b| b as i8).collect();
    off += L1W;

    let f32s = |n: usize, off: &mut usize| -> Vec<f32> {
        let v = bytes[*off..*off + n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        *off += n * 4;
        v
    };
    let l1b = f32s(OUTPUT_BUCKETS * L1_SIZE, &mut off);
    let l2w = f32s(OUTPUT_BUCKETS * L1_SIZE, &mut off);
    let l2b = f32s(OUTPUT_BUCKETS, &mut off);
    assert_eq!(off, PAYLOAD, "il parsing non ha consumato esattamente il payload");

    Network { l0w, l0b, l1w, l1b, l2w, l2b }
}

fn screlu_f32(x: f32) -> f32 {
    let y = x.clamp(0.0, 1.0);
    y * y
}

/// Accumulatori delle due prospettive + bucket di uscita, dalla FEN.
/// Identico a sanity.rs: e' la parte che le due architetture condividono.
pub fn accumulate(net_l0w: &[i16], net_l0b: &[i16], fen: &str) -> (Vec<i16>, Vec<i16>, usize) {
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

    let mut acc_us = net_l0b.to_vec();
    let mut acc_them = net_l0b.to_vec();
    for &(piece_type, is_black_piece, sq) in &pieces {
        for (acc, from_black_view, base, flip) in [
            (&mut acc_us, black_to_move, us_base, us_flip),
            (&mut acc_them, !black_to_move, them_base, them_flip),
        ] {
            let is_opp = is_black_piece != from_black_view;
            let sq_view = if from_black_view { sq ^ 56 } else { sq };
            let feat768 = usize::from(is_opp) * 384 + piece_type * 64 + sq_view;
            let feature = base + (feat768 ^ flip);
            for (a, w) in acc.iter_mut().zip(&net_l0w[feature * HIDDEN..(feature + 1) * HIDDEN]) {
                *a += *w;
            }
        }
    }

    let out_bucket = (pieces.len() - 2) / 32usize.div_ceil(OUTPUT_BUCKETS);
    (acc_us, acc_them, out_bucket)
}

/// CReLU + prodotto a coppie: 1024 valori i16 -> 512 valori in [0, QA].
fn pairwise(acc: &[i16], out: &mut [i32]) {
    let half = HIDDEN / 2;
    for j in 0..half {
        let a = i32::from(acc[j]).clamp(0, QA);
        let b = i32::from(acc[j + half]).clamp(0, QA);
        out[j] = a * b / QA;
    }
}

pub fn evaluate(net: &Network, fen: &str) -> i32 {
    let (acc_us, acc_them, ob) = accumulate(&net.l0w, &net.l0b, fen);

    let half = HIDDEN / 2;
    let mut hl1 = vec![0i32; HIDDEN];
    pairwise(&acc_us, &mut hl1[..half]);
    pairwise(&acc_them, &mut hl1[half..]);

    // l1: [L1_SIZE][HIDDEN] per bucket, gia' trasposto al salvataggio.
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

    let w2 = &net.l2w[ob * L1_SIZE..(ob + 1) * L1_SIZE];
    let mut y = net.l2b[ob];
    for (a, w) in a1.iter().zip(w2) {
        y += a * w;
    }
    (y * SCALE).round() as i32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("quantised.bin");
    let net = load(path);
    println!("{path}: layout OK ({PAYLOAD} B payload)");

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
        ("active kings endgame (w)", "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1"),
    ];

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

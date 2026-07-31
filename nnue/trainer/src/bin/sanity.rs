// Standalone (pure-std, no GPU) reader for bullet's quantised.bin — the
// REFERENCE IMPLEMENTATION for HydraY's engine-side loader/inference.
// Keep the C++ code byte-for-byte consistent with this.
//
// Net (HALFKA_PLAN.md): (768x4kb_hm -> HIDDEN)x2 -> OB, dual perspective,
// SCReLU, QA=255 QB=64 SCALE=400.
//  - 4 mirrored king input buckets (bullet ChessBucketsMirrored semantics):
//      for perspective X with own king on ksq_X (LERF, from X's own view):
//        flip_X   = 7 if file(ksq_X) > 3 else 0
//        bucket_X = KING_BUCKET_MAP[ksq_X]
//        feature  = 768*bucket_X + (feat768 ^ flip_X)
//      where feat768 = isOpp*384 + type*64 + sq_from_X (sq^56 for black view)
//      and `^ flip` only flips the file of the square component.
//  - OB = 8 material-count output buckets: (popcount(occ) - 2) / 4.
//
// File layout (little-endian i16, padded to a multiple of 64 B):
//   l0w: 4*768 columns x HIDDEN  (king-bucketed feature weights, QA,
//                                 factoriser already merged at save)
//   l0b: HIDDEN                  (accumulator bias, QA)
//   l1w: OB rows x 2*HIDDEN      (output weights, transposed, QB)
//   l1b: OB                      (output bias, QA*QB)
//
// Usage: cargo run -r --bin sanity -- <quantised.bin> [fen]...

const HIDDEN: usize = 512;
const INPUT_BUCKETS: usize = 4;
const OUTPUT_BUCKETS: usize = 8;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

// Keep in sync with trainer.rs BUCKET_LAYOUT and nnue/network.hpp.
// 32 entries: files a-d per rank, rank 1 first; e-h mirror onto d-a.
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

// bullet ChessBucketsMirrored::new expansion: 64-square map from the 32-entry
// half-board layout (files e-h mirror onto d-a).
fn king_bucket(ksq: usize) -> usize {
    const FILE_FOLD: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];
    BUCKET_LAYOUT[(ksq / 8) * 4 + FILE_FOLD[ksq % 8]]
}

struct Network {
    feature_weights: Vec<i16>, // [4*768 * HIDDEN], column f at f*HIDDEN..
    feature_bias: Vec<i16>,    // [HIDDEN]
    output_weights: Vec<i16>,  // [OB * 2 * HIDDEN], bucket b at b*2*HIDDEN..
    output_bias: Vec<i16>,     // [OB]
}

fn load(path: &str) -> Network {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let expected_i16s = INPUT_BUCKETS * 768 * HIDDEN + HIDDEN
        + OUTPUT_BUCKETS * 2 * HIDDEN + OUTPUT_BUCKETS;
    let expected_bytes = expected_i16s * 2;
    assert!(
        bytes.len() >= expected_bytes && bytes.len() % 64 == 0,
        "size mismatch: got {} bytes, expected {} (+pad to 64)",
        bytes.len(),
        expected_bytes
    );
    // bullet pads the file to 64 bytes with the repeating ASCII string
    // "bullet" — anything else in the tail means the layout drifted.
    assert!(
        bytes[expected_bytes..]
            .iter()
            .zip(b"bullet".iter().cycle())
            .all(|(a, b)| a == b),
        "unexpected padding tail — layout drift?"
    );

    let mut it = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]));
    let mut take = |n: usize| -> Vec<i16> { it.by_ref().take(n).collect() };

    let feature_weights = take(INPUT_BUCKETS * 768 * HIDDEN);
    let feature_bias = take(HIDDEN);
    let output_weights = take(OUTPUT_BUCKETS * 2 * HIDDEN);
    let output_bias = take(OUTPUT_BUCKETS);

    Network { feature_weights, feature_bias, output_weights, output_bias }
}

fn screlu(x: i16) -> i32 {
    let y = i32::from(x).clamp(0, QA);
    y * y
}

// Returns the stm-relative eval in centipawns.
fn evaluate(net: &Network, fen: &str) -> i32 {
    let mut fields = fen.split_whitespace();
    let board = fields.next().expect("empty FEN");
    let black_to_move = fields.next() == Some("b");

    // Pass 1: piece list (type, is_black, LERF sq) + king squares.
    let mut pieces: Vec<(usize, bool, usize)> = Vec::with_capacity(32);
    let (mut wk, mut bk) = (usize::MAX, usize::MAX);
    for (rank_from_top, row) in board.split('/').enumerate() {
        let rank = 7 - rank_from_top; // LERF rank
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

    // Per-perspective king bucket and horizontal flip, own king from own view.
    let params = |from_black_view: bool| {
        let own = if from_black_view { bk ^ 56 } else { wk };
        let flip = if own % 8 > 3 { 7 } else { 0 };
        (768 * king_bucket(own), flip)
    };
    let (us_base, us_flip) = params(black_to_move);
    let (them_base, them_flip) = params(!black_to_move);

    // Pass 2: accumulate both perspectives.
    let mut acc_us = net.feature_bias.clone();
    let mut acc_them = net.feature_bias.clone();
    for &(piece_type, is_black_piece, sq) in &pieces {
        for (acc, from_black_view, base, flip) in [
            (&mut acc_us, black_to_move, us_base, us_flip),
            (&mut acc_them, !black_to_move, them_base, them_flip),
        ] {
            let is_opp = is_black_piece != from_black_view;
            let sq_view = if from_black_view { sq ^ 56 } else { sq };
            let feat768 = usize::from(is_opp) * 384 + piece_type * 64 + sq_view;
            let feature = base + (feat768 ^ flip);
            for (a, w) in acc
                .iter_mut()
                .zip(&net.feature_weights[feature * HIDDEN..(feature + 1) * HIDDEN])
            {
                *a += *w;
            }
        }
    }

    // bullet MaterialCount<OB>: bucket from total piece count (incl. kings).
    let out_bucket = (pieces.len() - 2) / 32usize.div_ceil(OUTPUT_BUCKETS);
    let w = &net.output_weights[out_bucket * 2 * HIDDEN..(out_bucket + 1) * 2 * HIDDEN];

    let mut output = 0i32;
    for (a, wi) in acc_us.iter().zip(&w[..HIDDEN]) {
        output += screlu(*a) * i32::from(*wi);
    }
    for (a, wi) in acc_them.iter().zip(&w[HIDDEN..]) {
        output += screlu(*a) * i32::from(*wi);
    }
    output /= QA;
    output += i32::from(net.output_bias[out_bucket]);
    output * SCALE / (QA * QB)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("quantised.bin");
    let net = load(path);
    println!("{path}: layout OK");

    let default_fens = [
        ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("wN e4 extra (w)", "rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("bN e5 extra (b) [must equal previous]", "rnbqkbnr/pppppppp/8/4n3/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
        // Reduced-material refs: exercise different OUTPUT buckets.
        ("middlegame ~24 pieces (w)", "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"),
        ("KRPvKR (w)", "8/8/4k3/8/2r5/2K5/4P3/3R4 w - - 0 1"),
        ("KQvK (w)", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"),
        // King-placement refs: exercise INPUT buckets and the mirror fold.
        // Both castled short (mirror-fold both kings: g1/g8 -> b-file, bucket 0).
        ("both castled short (w)", "r4rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQ1RK1 w - - 0 1"),
        // White castled long (c1: bucket 1), black king still on e8 (bucket 1).
        ("white long castle vs e8 king (w)", "r3kb1r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPPBQPPP/2KR3R w kq - 0 1"),
        // Kings on 2nd rank (bucket 2) — pre/post-castling geometry gone.
        ("kings on 2nd rank (w)", "8/1k3ppp/1p6/p1p5/P1P5/1P4P1/1K3P1P/8 w - - 0 1"),
        // Active endgame kings in the center (bucket 3), mirrored files.
        ("active kings endgame (w)", "8/8/3k4/8/2p5/2P1K3/8/8 w - - 0 1"),
    ];

    let extra: Vec<(String, String)> = args[2..]
        .iter()
        .map(|f| (f.clone(), f.clone()))
        .collect();

    for (label, fen) in default_fens
        .iter()
        .map(|(l, f)| (l.to_string(), f.to_string()))
        .chain(extra)
    {
        println!("{label}: {} cp (stm)", evaluate(&net, &fen));
    }
}

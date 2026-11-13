// Threat inputs, pure std: the reference for nnue/threats.hpp, which must index
// every feature exactly as this file does.
//
// A feature is "piece a on square s attacks piece v on square t", with the
// real occupancy (a slider stops at its first blocker), for both colours of
// victim: attacks and defences. Everything is seen from one perspective:
// pieces 0-5 are its own P N B R Q K, 6-11 the opponent's, squares are in its
// view (own pawns move up) with the same file mirror as the piece features.
//
// Pairs that carry no information are not features:
//   - king attacks king (never legal);
//   - a attacks v whenever v attacks a, because a's attack set is a subset of
//     v's: P->B, P->Q, P->K, B->Q, R->Q, K->Q (the reverse feature is kept);
//   - two pieces of one type that attack each other (pawns: only of opposite
//     colours) are one feature, the one whose attacker square is lower.
// What is left: 70,050 features, indexed by attacker, victim, attacker square,
// then victim square in increasing order among the squares a could attack.

pub const FEATURES: usize = 70_050;
// Every piece attacks at most 8 others (one per ray or step), and at most 32
// pieces stand on the board.
pub const MAX_ACTIVE: usize = 256;
pub const NONE: u8 = 12;

const PAWN: usize = 0;
const KNIGHT: usize = 1;
const BISHOP: usize = 2;
const ROOK: usize = 3;
const QUEEN: usize = 4;
const KING: usize = 5;
const BACK_RANKS: u64 = 0xFF00_0000_0000_00FF;

const fn ray(sq: usize, occ: u64, dirs: &[(i32, i32)]) -> u64 {
    let mut out = 0u64;
    let mut d = 0;
    while d < dirs.len() {
        let (mut f, mut r) = ((sq % 8) as i32 + dirs[d].0, (sq / 8) as i32 + dirs[d].1);
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let bit = 1u64 << (r * 8 + f);
            out |= bit;
            if occ & bit != 0 {
                break;
            }
            f += dirs[d].0;
            r += dirs[d].1;
        }
        d += 1;
    }
    out
}

const fn steps(sq: usize, offsets: &[(i32, i32)]) -> u64 {
    let mut out = 0u64;
    let mut i = 0;
    while i < offsets.len() {
        let (f, r) = ((sq % 8) as i32 + offsets[i].0, (sq / 8) as i32 + offsets[i].1);
        if f >= 0 && f < 8 && r >= 0 && r < 8 {
            out |= 1u64 << (r * 8 + f);
        }
        i += 1;
    }
    out
}

const ORTHOGONAL: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const KNIGHT_STEPS: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
const KING_STEPS: [(i32, i32); 8] = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)];

/// The squares piece (0-11) on sq attacks with `occ` as the blockers. No pawn
/// stands on a back rank, so none attacks from there.
pub const fn attacks(piece: usize, sq: usize, occ: u64) -> u64 {
    match piece % 6 {
        PAWN if BACK_RANKS & (1u64 << sq) != 0 => 0,
        PAWN => steps(sq, if piece < 6 { &[(-1, 1), (1, 1)] } else { &[(-1, -1), (1, -1)] }),
        KNIGHT => steps(sq, &KNIGHT_STEPS),
        BISHOP => ray(sq, occ, &DIAGONAL),
        ROOK => ray(sq, occ, &ORTHOGONAL),
        QUEEN => ray(sq, occ, &DIAGONAL) | ray(sq, occ, &ORTHOGONAL),
        _ => steps(sq, &KING_STEPS),
    }
}

pub const fn pair_is_feature(a: usize, v: usize) -> bool {
    !matches!(
        (a % 6, v % 6),
        (KING, KING) | (PAWN, BISHOP) | (PAWN, QUEEN) | (PAWN, KING) | (BISHOP, QUEEN) | (ROOK, QUEEN) | (KING, QUEEN)
    )
}

const fn mutual(a: usize, v: usize) -> bool {
    a % 6 == v % 6 && (a % 6 != PAWN || a / 6 != v / 6)
}

const EMPTY_BOARD_ATTACKS: [[u64; 64]; 12] = {
    let mut table = [[0u64; 64]; 12];
    let mut i = 0;
    while i < 12 * 64 {
        table[i / 64][i % 64] = attacks(i / 64, i % 64, 0);
        i += 1;
    }
    table
};

/// The victim squares that index the features of a on `from` against v.
const fn targets(a: usize, v: usize, from: usize) -> u64 {
    let mut m = EMPTY_BOARD_ATTACKS[a][from];
    if v % 6 == PAWN {
        m &= !BACK_RANKS;
    }
    if mutual(a, v) {
        m &= !((2u64 << from).wrapping_sub(1));
    }
    m
}

const OFFSETS: ([[[u32; 64]; 12]; 12], usize) = {
    let mut offsets = [[[u32::MAX; 64]; 12]; 12];
    let mut next = 0usize;
    let mut a = 0;
    while a < 12 {
        let mut v = 0;
        while v < 12 {
            if pair_is_feature(a, v) {
                let mut from = 0;
                while from < 64 {
                    offsets[a][v][from] = next as u32;
                    next += targets(a, v, from).count_ones() as usize;
                    from += 1;
                }
            }
            v += 1;
        }
        a += 1;
    }
    (offsets, next)
};
const _: () = assert!(OFFSETS.1 == FEATURES);

pub fn index(a: usize, from: usize, v: usize, to: usize) -> Option<usize> {
    let offset = OFFSETS.0[a][v][from];
    if offset == u32::MAX || (mutual(a, v) && to < from) {
        return None;
    }
    let below = targets(a, v, from) & ((1u64 << to) - 1);
    Some(offset as usize + below.count_ones() as usize)
}

/// Every threat feature of one perspective, from its view of the board:
/// `board[sq]` is the piece on sq (0-11, or NONE) and `occ` the occupied squares.
pub fn map(board: &[u8; 64], occ: u64, mut f: impl FnMut(usize)) {
    let mut pieces = occ;
    while pieces != 0 {
        let from = pieces.trailing_zeros() as usize;
        pieces &= pieces - 1;
        let a = usize::from(board[from]);
        let mut victims = attacks(a, from, occ) & occ;
        while victims != 0 {
            let to = victims.trailing_zeros() as usize;
            victims &= victims - 1;
            if let Some(i) = index(a, from, usize::from(board[to]), to) {
                f(i);
            }
        }
    }
}

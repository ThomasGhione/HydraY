// Offline dataset stats — Fase 0 di DATAGEN_QUALITY_PLAN.md.
//
// Scansiona file bulletformat (32 B/record) e riporta:
//  - anomalie strutturali per file: coda parziale (byte non multipli di 32),
//    record azzerati (occ == 0: power-loss a metà scrittura);
//  - dup-rate globale ESATTO sulla chiave 24-byte (occ+pcs, stm-relative,
//    come da piano) via hash u64 FNV-1a + sort (collisioni attese ~0 a 500M);
//  - top-10 conteggi delle posizioni più ripetute;
//  - istogramma |score| e distribuzione W/D/L (result stm-relative).
//
// Pure std, niente GPU. Uso:
//   cargo run -r --bin datastats -- <file.bin> [file2.bin ...]
//
// RAM: 8 B per record (Vec<u64> degli hash) — ~4 GB a 500M record.

use std::io::Read;

const REC: usize = 32;
const KEY_BYTES: usize = 24; // occ (8) + pcs (16)

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "uso: datastats <file.bin> [...]");

    let mut hashes: Vec<u64> = Vec::new();
    let mut total: u64 = 0;
    let mut zeroed: u64 = 0;
    let mut wdl = [0u64; 3];
    // |score| in cp: [0,50), [50,100), [100,200), [200,400), [400,800), [800,3000], >3000
    let mut score_hist = [0u64; 7];

    for path in &files {
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("apertura {path}: {e}"))
            .read_to_end(&mut buf)
            .unwrap_or_else(|e| panic!("lettura {path}: {e}"));

        let tail = buf.len() % REC;
        let n = buf.len() / REC;
        let mut file_zeroed = 0u64;

        hashes.reserve(n);
        for rec in buf.chunks_exact(REC) {
            if rec[..8] == [0u8; 8] {
                file_zeroed += 1;
                continue; // non conteggiato nelle stats: verrà strippato
            }
            hashes.push(fnv1a64(&rec[..KEY_BYTES]));

            let score = i16::from_le_bytes([rec[24], rec[25]]);
            let a = i32::from(score).unsigned_abs();
            let bucket = match a {
                0..=49 => 0, 50..=99 => 1, 100..=199 => 2, 200..=399 => 3,
                400..=799 => 4, 800..=3000 => 5, _ => 6,
            };
            score_hist[bucket as usize] += 1;

            let r = rec[26] as usize;
            if r < 3 { wdl[r] += 1; }
        }
        total += n as u64;
        zeroed += file_zeroed;
        if tail != 0 || file_zeroed != 0 {
            println!("ANOMALIA {path}: coda {tail} B, {file_zeroed} record azzerati");
        }
    }

    let valid = hashes.len() as u64;
    println!("record totali : {total}");
    println!("azzerati      : {zeroed}");
    println!("validi        : {valid}");

    hashes.sort_unstable();
    let mut dups: u64 = 0;
    let mut run_lengths: Vec<u64> = Vec::new();
    let mut i = 0usize;
    while i < hashes.len() {
        let mut j = i + 1;
        while j < hashes.len() && hashes[j] == hashes[i] { j += 1; }
        let run = (j - i) as u64;
        if run > 1 {
            dups += run - 1;
            run_lengths.push(run);
        }
        i = j;
    }
    run_lengths.sort_unstable_by(|a, b| b.cmp(a));
    let top: Vec<u64> = run_lengths.iter().take(10).copied().collect();

    println!("duplicati     : {dups} ({:.2}%)", 100.0 * dups as f64 / valid as f64);
    println!("top-10 run    : {top:?}");
    println!("W/D/L stm     : {:.1}% / {:.1}% / {:.1}%",
        100.0 * wdl[2] as f64 / valid as f64,
        100.0 * wdl[1] as f64 / valid as f64,
        100.0 * wdl[0] as f64 / valid as f64);
    let labels = ["0-49", "50-99", "100-199", "200-399", "400-799", "800-3000", ">3000!"];
    print!("|score| cp    : ");
    for (l, c) in labels.iter().zip(&score_hist) {
        print!("{l}: {:.1}%  ", 100.0 * *c as f64 / valid as f64);
    }
    println!();
}

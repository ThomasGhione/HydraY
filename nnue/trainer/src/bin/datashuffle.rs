// Global shuffle esterno per dataset bulletformat (32 B/record).
//
// Due passate, RAM O(dataset/K):
//   1. streaming di tutti gli input: ogni record va in uno di K shard
//      scelto uniformemente (splitmix64, seed fisso -> riproducibile);
//      i record azzerati (occ == 0) vengono strippati qui.
//   2. ogni shard viene caricato, permutato in RAM (Fisher-Yates) e
//      appeso all'output.
// Il risultato è una permutazione uniforme globale (shuffle a shard).
//
// Uso: cargo run -r --bin datashuffle -- <out.bin> <in1.bin> [in2.bin ...]
// Gli shard temporanei (<out>.shard<N>) vengono rimossi a fine run.

use std::io::{BufWriter, Read, Write};

const REC: usize = 32;
const SHARDS: usize = 64;
const SEED: u64 = 0x48594452_41593333; // "HYDRAY33"

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 2, "uso: datashuffle <out.bin> <in.bin> [...]");
    let out_path = &args[0];
    let inputs = &args[1..];

    let mut rng = SplitMix64(SEED);
    let shard_path = |i: usize| format!("{out_path}.shard{i}");

    // Pass 1: scatter.
    let mut shards: Vec<BufWriter<std::fs::File>> = (0..SHARDS)
        .map(|i| BufWriter::with_capacity(1 << 20,
            std::fs::File::create(shard_path(i)).expect("crea shard")))
        .collect();
    let mut total: u64 = 0;
    let mut stripped: u64 = 0;
    for path in inputs {
        let mut buf = Vec::new();
        std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("apertura {path}: {e}"))
            .read_to_end(&mut buf)
            .unwrap_or_else(|e| panic!("lettura {path}: {e}"));
        let usable = buf.len() - buf.len() % REC;
        for rec in buf[..usable].chunks_exact(REC) {
            if rec[..8] == [0u8; 8] { stripped += 1; continue; }
            let s = (rng.next() % SHARDS as u64) as usize;
            shards[s].write_all(rec).expect("scrittura shard");
            total += 1;
        }
    }
    for w in &mut shards { w.flush().expect("flush shard"); }
    drop(shards);
    println!("pass 1: {total} record in {SHARDS} shard ({stripped} azzerati strippati)");

    // Pass 2: shuffle per shard + concat.
    let mut out = BufWriter::with_capacity(1 << 22,
        std::fs::File::create(out_path).expect("crea output"));
    for i in 0..SHARDS {
        let mut buf = Vec::new();
        std::fs::File::open(shard_path(i)).expect("apertura shard")
            .read_to_end(&mut buf).expect("lettura shard");
        let n = buf.len() / REC;
        // Fisher-Yates sugli indici dei record.
        for k in (1..n).rev() {
            let j = (rng.next() % (k as u64 + 1)) as usize;
            if j != k {
                for b in 0..REC {
                    buf.swap(k * REC + b, j * REC + b);
                }
            }
        }
        out.write_all(&buf).expect("scrittura output");
        std::fs::remove_file(shard_path(i)).expect("rimozione shard");
    }
    out.flush().expect("flush output");
    println!("pass 2: shuffle completato -> {out_path}");
}

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
// Il pass 1 streama a blocchi di 16 MiB: la RAM non dipende dalla taglia dei
// singoli input.
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
    // Lo scatter STREAMA a blocchi: leggere un input intero in RAM costava
    // O(file piu grande), non O(dataset/K) come dice l'intestazione. Finche
    // gli input erano gli shard grezzi da pochi GB non si notava; un singolo
    // prefisso da 37 GB ha fatto scattare l'OOM killer su una macchina da 30.
    const CHUNK: usize = 1 << 24; // 16 MiB, multiplo di REC
    let mut buf = vec![0u8; CHUNK];
    for path in inputs {
        let mut f = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("apertura {path}: {e}"));
        // Byte di coda che non completano un record: restano in testa al buffer
        // per il giro successivo, perche read() puo tornare meno di CHUNK.
        let mut carry = 0usize;
        loop {
            let n = f.read(&mut buf[carry..])
                .unwrap_or_else(|e| panic!("lettura {path}: {e}"));
            if n == 0 { break; }
            let avail = carry + n;
            let usable = avail - avail % REC;
            for rec in buf[..usable].chunks_exact(REC) {
                if rec[..8] == [0u8; 8] { stripped += 1; continue; }
                let s = (rng.next() % SHARDS as u64) as usize;
                shards[s].write_all(rec).expect("scrittura shard");
                total += 1;
            }
            buf.copy_within(usable..avail, 0);
            carry = avail - usable;
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

// Replaces a slice of a shuffled bulletformat dataset with records from
// another, for data A/B tests whose arms must differ in nothing else.
//
// Usage: datamix <out.bin> <base|-> <base_records> <repl|-> <repl_records> <seed>
//
// The output holds base_records records: repl_records from <repl>, the rest
// the leading records of <base>. The slots that take a replacement are drawn
// by selection sampling, so there are exactly repl_records of them, spread
// uniformly; with both inputs shuffled, the output is shuffled too. The slot
// pattern depends only on the counts and the seed, so two arms built with the
// same arguments keep the same base records in the same places.
//
// Both inputs are read to the end and must hold exactly the stated number of
// records: a truncated decompression stream ends early with no error of its
// own. The base records past the output are read and dropped.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

const REC: u64 = 32;

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

fn open(path: &str) -> BufReader<Box<dyn Read>> {
    let inner: Box<dyn Read> = if path == "-" {
        Box::new(io::stdin())
    } else {
        Box::new(File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}")))
    };
    BufReader::with_capacity(1 << 22, inner)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(a.len() == 6,
        "usage: datamix <out.bin> <base|-> <base_records> <repl|-> <repl_records> <seed>");
    let num = |s: &str| s.parse::<u64>().unwrap_or_else(|_| panic!("not a count: {s}"));
    let (n, r, seed) = (num(&a[2]), num(&a[4]), num(&a[5]));
    assert!(r <= n, "more replacement records than output slots");

    let mut base = open(&a[1]);
    let mut repl = open(&a[3]);
    let mut out = BufWriter::with_capacity(1 << 22,
        File::create(&a[0]).unwrap_or_else(|e| panic!("create {}: {e}", a[0])));

    let mut rng = SplitMix64(seed);
    let mut rec = [0u8; REC as usize];
    let mut need = r;
    for left in (1..=n).rev() {
        // Replacement with probability need / left: multiply-high maps the
        // 64-bit draw onto [0, left) without a division.
        let take = ((rng.next() as u128 * left as u128) >> 64) < need as u128;
        let src = if take { need -= 1; &mut repl } else { &mut base };
        src.read_exact(&mut rec).expect("input ended before the output was full");
        out.write_all(&rec).expect("write output");
    }
    out.flush().expect("flush output");

    let base_rest = io::copy(&mut base, &mut io::sink()).expect("read base");
    let repl_rest = io::copy(&mut repl, &mut io::sink()).expect("read replacement");
    assert!(base_rest == r * REC, "base has {base_rest} bytes left, expected {}", r * REC);
    assert!(repl_rest == 0, "replacement has {repl_rest} bytes left, expected 0");
    eprintln!("datamix: {n} records = {} base + {r} replacement, {r} base dropped", n - r);
}

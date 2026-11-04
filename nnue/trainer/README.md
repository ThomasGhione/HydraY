# HydraY NNUE trainer (bullet)

Rete v1: `(768 -> 256)x2 -> 1`, dual perspective, SCReLU, QA=255 QB=64, SCALE=400.

## Pipeline

1. **Dati** (laptop): `./chess datagen` → `nnue/data/hydray.sNNNN.bin` (bulletformat, shard chiusi).
   Snapshot + merge:
   ```sh
   # gli shard chiusi sono completi: solo un .part va troncato a multipli di 32 B
   bullet-utils interleave hydray.s0000.bin hydray.s0001.bin ... --output hydray_interleaved.bin
   bullet-utils validate --input hydray_interleaved.bin
   ```
2. **Training** (Colab T4, gratis): apri `colab_v1.ipynb` su Colab,
   runtime GPU, esegui le celle. Oppure a mano su una macchina CUDA:
   ```sh
   cargo run -r --bin trainer --features cuda -- <data.bin> [superbatches] [net_id]
   ```
   Shakedown ≈ 10 superbatch; run v1 completa ≈ 40 (≈100M sample/superbatch).
3. **Verifica** (qualsiasi macchina, niente GPU):
   ```sh
   cargo run -r --bin sanity -- checkpoints/<id>-<N>/quantised.bin
   ```
   Attese: startpos ≈ +20..50 cp; la coppia specchiata identica; ±donna enorme.

`sanity.rs` è la **specifica di riferimento** del formato `quantised.bin` per il
loader/inferenza C++ engine-side (Fase 3): tenerli coerenti byte-per-byte.

bullet è pinnato per rev in `Cargo.toml`; bump consapevoli (il layout di
`quantised.bin` deve restare quello che sanity.rs valida).

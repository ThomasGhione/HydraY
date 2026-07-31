// HydraY NNUE v4 trainer (HALFKA_PLAN.md: HalfKA + king bucket).
//
// Architecture: (768x8kb_hm -> 512)x2 -> 8ob — 8 mirrored king buckets on the
// inputs (bullet ChessBucketsMirrored: feature = 768*bucket[ksq] + (feat768 ^
// flip), flip = 7 iff file(ksq) > d), material-count output buckets as in the
// ob cycle, SCReLU, QA=255/QB=64. Input-layer factoriser l0f (shared 768xH
// component summed onto every bucket) — merged into l0w at save time via
// SavedFormat::transform, so quantised.bin needs no factoriser awareness.
// Weight clipping tightened to ±0.99 on l0w/l0f: the SAVED weight is the sum
// of the two, and |sum| must stay <= ~1.98 for i16 quantisation at QA=255.
// Adapted from bullet's examples/progression/3_input_buckets.rs (pinned rev).
//
// Usage: cargo run -r --bin trainer --features cuda -- <data.bin> [superbatches]
//                                                       [net_id] [start_sb] [resume_ckpt]
//   shakedown: ~10 superbatches; full run: ~40.
//
// STAGED TRAINING (start_sb + resume_ckpt) exists because the free Colab
// runtime has ~66 GB of local disk while the full dataset is 88 GB, and the
// Drive mount caches locally everything it reads — so streaming a 40-superbatch
// run fills the disk around superbatch 28. Instead, train on one slice at a
// time: each stage exits (releasing the cache), the local file is replaced with
// the next slice, and training resumes from the checkpoint. The net ends up
// having seen the whole dataset with no slice ever exceeding the disk.
//
// Stage boundaries must land on multiples of `save_rate` (10 below), since a
// stage can only resume from a checkpoint that was actually written. Four
// slices of a quarter of the dataset each therefore fit the 40-superbatch
// schedule exactly, and 10 superbatches consume 1B samples against a 688M
// slice — the same 1.45 epochs the whole run would do over the whole dataset:
//
//   stage 1:  <slice A> 40 hydray-x
//   stage 2:  <slice B> 40 hydray-x 11 checkpoints/hydray-x-10
//   stage 3:  <slice C> 40 hydray-x 21 checkpoints/hydray-x-20
//   stage 4:  <slice D> 40 hydray-x 31 checkpoints/hydray-x-30
//
// Two properties of bullet make this sound, both verified in its source rather
// than assumed: `Step::new` iterates from `start_superbatch`, so `StepLR` sees
// ABSOLUTE indices and the learning-rate drop still lands where the schedule
// says; and `load_from_checkpoint` restores `optimiser_state`, so AdamW's
// moments carry across a stage boundary instead of restarting cold.
//
// TEST_PATH (env var) points at a held-out slice and turns on validation loss.
// Without it only training loss is reported, which cannot show overfitting —
// the 8-bucket map had a BETTER training loss than the 4-bucket one while
// playing 12 Elo worse, and that was only diagnosable by elimination.

use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::{LocalSettings, TestDataset},
    },
    value::{loader, ValueTrainerBuilder},
};

const HIDDEN_SIZE: usize = 512;
const OUTPUT_BUCKETS: usize = 8;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

// Keep in sync with nnue/network.hpp KING_BUCKET_MAP and sanity.rs.
// 32 entries: files a-d per rank, rank 1 first; e-h mirror onto d-a.
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
];
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../data/hydray_16M_interleaved.bin".to_string());
    let superbatches: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let net_id = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "hydray-halfka-shakedown".to_string());
    // Absolute index of the first superbatch of THIS stage; `superbatches` stays
    // the total for the whole run so the learning-rate schedule is unchanged.
    let start_superbatch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let resume_from = args.get(5).cloned();
    let test_path = std::env::var("TEST_PATH").ok();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        // Quantised layout consumed by sanity.rs and the engine loader:
        // l0w (4x768 cols x H, QA, factoriser merged) | l0b (H, QA) |
        // l1w (OBx2H, QB, transposed: each bucket contiguous) | l1b (OB, QA*QB)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // Saved l0 weight = l0w + l0f: clip both to ±0.99 so the merged value
    // stays inside the i16-safe ±1.98 at QA=255.
    let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter);
    trainer.optimiser.set_params_for_weight("l0f", stricter);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104, // ~100M samples per superbatch
            start_superbatch,
            end_superbatch: superbatches,
        },
        // NNUE_PLAN lambda = 0.7 on the search score; bullet's `wdl` weights the
        // game RESULT, so wdl = 1 - lambda = 0.3.
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.1,
            step: (superbatches / 2).max(1),
        },
        // Intermediate checkpoints: a Colab session can die mid-run; a sb-30
        // checkpoint is still a usable net, a lost 4h run is not.
        save_rate: superbatches.clamp(1, 10),
    };

    let settings = LocalSettings {
        threads: 2,
        test_set: test_path.as_deref().map(TestDataset::at),
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data_path.as_str()]);

    // Resume before run(): load_from_checkpoint restores the optimiser state,
    // so AdamW's moments survive the stage boundary. Loading after run() would
    // silently train from scratch on this slice.
    if let Some(ckpt) = &resume_from {
        println!("resuming from {ckpt} at superbatch {start_superbatch}");
        trainer.load_from_checkpoint(ckpt);
    }

    trainer.run(&schedule, &settings, &data_loader);

    // Post-training sanity (values in cp-ish units via eval_scale):
    //  - startpos should land around +20..50 for the side to move
    //  - the mirrored pair must print (near-)identical values
    //  - the up-a-queen positions must be hugely positive/negative
    println!("=== sanity evals (stm-relative) ===");
    for (label, fen) in [
        ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        // Knight developed off b1/b8 rather than added: an extra piece leaves
        // the side with 17 men, which bullet's FEN parser rejects with a panic
        // that killed the whole sanity block after training had finished.
        ("wN on e4 (w)", "rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/R1BQKBNR w - - 0 1"),
        ("bN on e5 (b) [mirror of previous]", "r1bqkbnr/pppppppp/8/4n3/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
    ] {
        println!("{label}: {:.1}", trainer.eval(fen));
    }
}

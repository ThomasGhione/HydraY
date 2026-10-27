// HydraY NNUE: the deep net, (768x4kb_hm -> 1024)x2 -> pairwise -> 1024 -> 16 -> 1,
// with material-count output buckets on BOTH layers after the feature transformer.
//
// One hidden layer, not bullet's 16 -> 32 -> 1 example: the second layer costs
// little but was never validated on a net this size. Adapted from bullet
// examples/progression/4_multi_layer.rs, the sequel of 3_input_buckets.rs that
// trainer.rs comes from.
//
// Two choices come with the architecture and are not free:
//   1. CReLU on the feature transformer, with the pairwise product of its two
//      halves supplying the quadratic term and halving the width (1024 -> 512 per
//      perspective), so l1's input is 1024 and not 2048;
//   2. init_with_effective_input_size(32): fan-in 32 (the active features per
//      position), the right init for sparse inputs.
//
// OUTPUT BUCKETS: 16 by default, 8 for the control arm of the 16-bucket test.
// bullet's MaterialCount takes the count as a const generic, so it is read at
// COMPILE time from HYDRAY_OUTPUT_BUCKETS; cargo rebuilds when the variable
// changes, so EVERY cargo command of one run must pass the same value.
//
// QUANTISED FILE LAYOUT (little-endian, save_format order), OB = output buckets:
//
//   l0w  [4*768][1024]  i16  QA=255   (factoriser folded in)
//   l0b  [1024]         i16  QA=255
//   l1w  [OB][16][1024] i8   QB=64    (transposed: each bucket contiguous)
//   l1b  [OB][16]       f32
//   l2w  [OB][1][16]    f32           (transposed)
//   l2b  [OB]           f32
//
//   OB = 16: 6,557,760 B (a multiple of 64: bullet adds no padding)
//   OB = 8:  6,425,632 B, padded to 6,425,664
//
// Only l0 and l1w are quantised; the 16 -> 1 tail stays float. sanity_deep.rs is
// the reference reader, and the C++ must agree with it byte for byte.
//
// Usage, stages included:
//   HYDRAY_OUTPUT_BUCKETS=16 cargo run -r --bin trainer_deep --features cuda --
//       <data.bin[,data2.bin...]> [superbatches] [net_id] [start_sb] [resume_ckpt]
//   Several data files are read as one stream, in order. A resumed run skips to
//   the sample where the previous stage stopped (bullet counts the batches
//   done, modulo the epoch), so stages train exactly like one uninterrupted run.
//   STAGE_END=<n> ends the current stage; it must sit right in front of cargo
//   (`env STAGE_END=40 cargo ...`), see trainer.rs.

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
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

const HIDDEN_SIZE: usize = 1024;
// Width of the hidden layer; the forward's cost scales linearly with it.
const L1_SIZE: usize = 16;
const OUTPUT_BUCKETS: usize = match option_env!("HYDRAY_OUTPUT_BUCKETS") {
    None => 16,
    Some(s) => match s.as_bytes() {
        b"16" => 16,
        b"8" => 8,
        _ => panic!("HYDRAY_OUTPUT_BUCKETS must be 8 or 16"),
    },
};
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

// Keep in sync with nnue/network.hpp KING_BUCKET_MAP and sanity.rs.
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
        .unwrap_or_else(|| "hydray-deep-shakedown".to_string());
    let start_superbatch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let resume_from = args.get(5).cloned();
    let stage_end: usize = std::env::var("STAGE_END")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(superbatches);
    // The notebook checks this line against the arm it means to train.
    println!("trainer_deep: {HIDDEN_SIZE} -> {L1_SIZE} -> 1, {OUTPUT_BUCKETS} output buckets");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // i8 at QB=64: bullet FAILS on a weight that does not fit (it does
            // not truncate silently), so a wrong scale shows up at save time.
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(QB),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);

            let l1 = builder.new_affine("l1", HIDDEN_SIZE, OUTPUT_BUCKETS * L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, OUTPUT_BUCKETS);

            // Pairwise: CReLU on the two halves of the feature transformer, then
            // their product. A fast form of `.crelu().pairwise_mul()`.
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, HIDDEN_SIZE / 2) * ft(stm_inputs, HIDDEN_SIZE / 2, HIDDEN_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, HIDDEN_SIZE / 2) * ft(ntm_inputs, HIDDEN_SIZE / 2, HIDDEN_SIZE);

            // l1 computes every bucket's outputs and select keeps one: its cost
            // grows with OUTPUT_BUCKETS.
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            l2.forward(hl2).select(output_buckets)
        });

    // The SAVED l0 weight is l0w + l0f: clipping both to ±0.99 keeps the sum in
    // the ±1.98 that is i16-safe at QA=255.
    let stricter = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter);
    trainer.optimiser.set_params_for_weight("l0f", stricter);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch,
            end_superbatch: stage_end,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.1,
            step: (superbatches / 2).max(1),
        },
        save_rate: superbatches.clamp(1, 10),
    };

    let settings = LocalSettings {
        threads: 2,
        test_set: None, // not implemented by bullet at the pinned rev: see trainer.rs
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_paths: Vec<&str> = data_path.split(',').collect();
    let data_loader = loader::DirectSequentialDataLoader::new(&data_paths);

    if let Some(ckpt) = &resume_from {
        println!("resuming from {ckpt} at superbatch {start_superbatch}");
        trainer.load_from_checkpoint(ckpt);
    }

    trainer.run(&schedule, &settings, &data_loader);

    println!("=== sanity evals (stm-relative) ===");
    for (label, fen) in [
        ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
        ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
        ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
    ] {
        println!("{label}: {}", trainer.eval(fen));
    }
}

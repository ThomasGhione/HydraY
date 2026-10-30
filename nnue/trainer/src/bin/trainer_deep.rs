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
//
// RECIPE: read from the environment at run time, so one build trains every arm of
// a recipe test. Each variable left unset keeps the recipe of the 4.2.0 net, and
// every stage of a run must see the same values (the notebook passes them).
//   HYDRAY_LR=step|cosine   step: x0.1 at superbatches/2 (default); cosine: down to
//                           HYDRAY_LR_FINAL at the last superbatch
//   HYDRAY_LR_START         default 0.001
//   HYDRAY_LR_FINAL         cosine only, default HYDRAY_LR_START * 0.3^5 (bullet's examples)
//   HYDRAY_WARMUP           batches of LR warmup in superbatch 1 (bullet's lr::Warmup), default 0
//   HYDRAY_WDL              weight of the game result in the target, default 0.3
//   HYDRAY_EVAL_SCALE       sigmoid scale of the labels, default 400
//   HYDRAY_LOSS=mse|pow2.5  default mse
//   HYDRAY_L1B_INIT         initial l1 bias, default 0 (bullet zeroes it)
//   HYDRAY_FLOAT_CLIP       weight clip of the float tail l1b, l2w, l2b; default
//                           bullet's 1.98, which only the quantised layers need
//   HYDRAY_INIT_NET         start from this quantised net (same layout) instead of
//                           random weights; the run checks it evaluates the same
//   HYDRAY_REVIVE=1         with HYDRAY_INIT_NET: re-initialise its dead l1 neurons
//                           (bias HYDRAY_L1B_INIT, default 0.1), see sanity_deep.rs
//   HYDRAY_QAT=1            quantisation-aware training: the forward computes what
//                           the quantised net computes, see QAT below
//
// QAT: bullet's faux_quantise puts a value on the grid of step 1/q in the forward
// and passes the gradient through unchanged (straight-through). Four places, the
// four roundings of network_deep.hpp: l0's saved weight (l0w + l0f, which is what
// the save rounds) and l0b to 1/QA, the pairwise product truncated to 1/QA (the
// engine's integer division, nudged by half a step of the float error margin),
// l1w to 1/QB. The tail stays float. Measured on antidead-160sb, 600k holdout
// positions, float checkpoint against the engine: l1w rounding 14.8 cp rms, l0
// 3.9, pairwise truncation 2.6 (nnue/tools/qatgap.cpp).

use bullet_compiler::tensor::TValue;
use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        Affine, InitSettings, ModelBuilder, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, lr::LrScheduler, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

// The reference reader: it loads HYDRAY_INIT_NET and converts it to bullet's layout.
#[allow(dead_code)]
#[path = "sanity_deep.rs"]
mod sanity_deep;

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

const SANITY_FENS: [(&str, &str); 5] = [
    ("startpos (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("startpos (b)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
    ("stm up a queen (w)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("stm down a queen (w)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w - - 0 1"),
    ("middlegame ~24 pieces (w)", "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 1"),
];

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| panic!("{name}={v} does not parse")),
        Err(_) => default,
    }
}

// Both delegate to bullet: Step is the 4.2.0 recipe, Cosine bullet's own default
// for this architecture (examples/progression/4_multi_layer.rs).
#[derive(Clone, Debug)]
enum Lr {
    Step(lr::StepLR),
    Cosine(lr::CosineDecayLR),
}

impl LrScheduler for Lr {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        match self {
            Lr::Step(s) => s.lr(batch, superbatch),
            Lr::Cosine(c) => c.lr(batch, superbatch),
        }
    }

    fn colourful(&self) -> String {
        match self {
            Lr::Step(s) => s.colourful(),
            Lr::Cosine(c) => c.colourful(),
        }
    }
}

// bullet's new_affine, with the initial values given.
fn affine<'a>(b: &'a ModelBuilder, id: &str, inputs: usize, outputs: usize, w: InitSettings, bias: InitSettings) -> Affine<'a> {
    Affine { weights: b.new_weights(format!("{id}w"), (outputs, inputs), w), bias: b.new_weights(format!("{id}b"), (outputs, 1), bias) }
}

fn custom(values: &[f32]) -> InitSettings {
    InitSettings::Custom(TValue::F32(values.to_vec()))
}

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

    let lr_kind: String = env_or("HYDRAY_LR", "step".to_string());
    let lr_start: f32 = env_or("HYDRAY_LR_START", 0.001);
    let lr_final: f32 = env_or("HYDRAY_LR_FINAL", lr_start * 0.3f32.powi(5));
    let warmup: usize = env_or("HYDRAY_WARMUP", 0);
    let wdl_weight: f32 = env_or("HYDRAY_WDL", 0.3);
    let eval_scale: f32 = env_or("HYDRAY_EVAL_SCALE", SCALE as f32);
    let loss: String = env_or("HYDRAY_LOSS", "mse".to_string());
    let float_clip: f32 = env_or("HYDRAY_FLOAT_CLIP", AdamWParams::default().max_weight);
    let init_net: String = env_or("HYDRAY_INIT_NET", String::new());
    let revive = env_or("HYDRAY_REVIVE", 0) != 0;
    let l1b_init: f32 = env_or("HYDRAY_L1B_INIT", if revive { 0.1 } else { 0.0 });
    let qat = env_or("HYDRAY_QAT", 0) != 0;
    assert!(!revive || !init_net.is_empty(), "HYDRAY_REVIVE needs HYDRAY_INIT_NET");
    let lr_scheduler = match lr_kind.as_str() {
        "step" => Lr::Step(lr::StepLR { start: lr_start, gamma: 0.1, step: (superbatches / 2).max(1) }),
        "cosine" => Lr::Cosine(lr::CosineDecayLR { initial_lr: lr_start, final_lr: lr_final, final_superbatch: superbatches }),
        other => panic!("HYDRAY_LR={other}: step or cosine"),
    };
    let pow25 = match loss.as_str() {
        "mse" => false,
        "pow2.5" => true,
        other => panic!("HYDRAY_LOSS={other}: mse or pow2.5"),
    };
    // One line with the whole recipe: the notebook checks it against the arm.
    println!(
        "recipe: lr {lr_kind} {lr_start} (final {lr_final}), warmup {warmup}, wdl {wdl_weight}, eval_scale {eval_scale}, \
         loss {loss}, l1b_init {l1b_init}, float_clip {float_clip}, init_net '{init_net}', revive {revive}, qat {qat}"
    );

    let init = (!init_net.is_empty()).then(|| {
        let net = sanity_deep::load(&init_net);
        assert_eq!(net.ob, OUTPUT_BUCKETS, "{init_net}: {} output buckets, this build has {OUTPUT_BUCKETS}", net.ob);
        let b = sanity_deep::to_bullet(&net, revive.then_some(l1b_init));
        println!("init net {init_net}: {} dead l1 neurons revived, {} l0 values past the ±0.99 clip", b.revived, b.clipped);
        b
    });

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
        .loss_fn(if pow25 {
            |output, target| output.sigmoid().power_error(target, 2.5)
        } else {
            |output, target| output.sigmoid().squared_error(target)
        })
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Same weights, names and order as bullet's new_affine: unset knobs give
            // the same initial net as before.
            let l0f_init = init.as_ref().map_or(InitSettings::Zeroed, |n| custom(&n.l0f));
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), l0f_init);
            let mut l0 = match &init {
                Some(n) => affine(builder, "l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE, custom(&n.l0w), custom(&n.l0b)),
                None => {
                    let l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN_SIZE);
                    l0.init_with_effective_input_size(32);
                    l0
                }
            };
            l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);
            let (qa, qb) = (QA as f32, QB as f32);
            if qat {
                l0.weights = l0.weights.faux_quantise(qa, true);
                l0.bias = l0.bias.faux_quantise(qa, true);
            }

            // l1's inputs are all >= 0, so with a zero bias a neuron can fall below
            // zero for every input and never fire again: HYDRAY_L1B_INIT.
            let (l1w_init, l1_bias) = match &init {
                Some(n) => (custom(&n.l1w), custom(&n.l1b)),
                None => (
                    InitSettings::Normal { mean: 0.0, stdev: (2.0 / HIDDEN_SIZE as f32).sqrt() },
                    if l1b_init == 0.0 { InitSettings::Zeroed } else { custom(&[l1b_init; OUTPUT_BUCKETS * L1_SIZE]) },
                ),
            };
            let mut l1 = affine(builder, "l1", HIDDEN_SIZE, OUTPUT_BUCKETS * L1_SIZE, l1w_init, l1_bias);
            if qat {
                l1.weights = l1.weights.faux_quantise(qb, true);
            }
            let l2 = match &init {
                Some(n) => affine(builder, "l2", L1_SIZE, OUTPUT_BUCKETS, custom(&n.l2w), custom(&n.l2b)),
                None => builder.new_affine("l2", L1_SIZE, OUTPUT_BUCKETS),
            };

            // Pairwise: CReLU on the two halves of the feature transformer, then
            // their product. A fast form of `.crelu().pairwise_mul()`.
            // Under QAT the product is truncated to 1/QA, as the engine's integer
            // division; the nudge keeps float error from truncating a whole value
            // one step low (sanity_deep.rs, PAIRWISE_NUDGE).
            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let pairwise = |input| {
                let p = ft(input, 0, HIDDEN_SIZE / 2) * ft(input, HIDDEN_SIZE / 2, HIDDEN_SIZE);
                if qat { (p + sanity_deep::PAIRWISE_NUDGE).faux_quantise(qa, false) } else { p }
            };
            let stm_hidden = pairwise(stm_inputs);
            let ntm_hidden = pairwise(ntm_inputs);

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
    let float_tail = AdamWParams { max_weight: float_clip, min_weight: -float_clip, ..Default::default() };
    for id in ["l1b", "l2w", "l2b"] {
        trainer.optimiser.set_params_for_weight(id, float_tail);
    }

    let schedule = TrainingSchedule {
        net_id,
        eval_scale,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch,
            end_superbatch: stage_end,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        lr_scheduler: lr::Warmup { inner: lr_scheduler, warmup_batches: warmup },
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
    } else if let Some(n) = &init {
        // Against the float forward over the same weights (under QAT, the same
        // roundings): a wrong layout misses by hundreds of centipawns, summation
        // order by far less than one.
        for (label, fen) in SANITY_FENS {
            let ours = trainer.eval(fen) * SCALE as f32;
            let reference = sanity_deep::evaluate_bullet(n, fen, qat);
            println!("init check {label}: bullet {ours:.1} cp, reference {reference:.1} cp");
            assert!((ours - reference).abs() < 1.0, "the starting net does not evaluate like {init_net}");
        }
    }

    trainer.run(&schedule, &settings, &data_loader);

    println!("=== sanity evals (stm-relative) ===");
    for (label, fen) in SANITY_FENS {
        println!("{label}: {}", trainer.eval(fen));
    }
}

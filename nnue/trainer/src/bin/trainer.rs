// HydraY NNUE v3 trainer (NNUE_PLAN.md, ciclo v3: output buckets).
//
// Architecture: (768 -> 512)x2 -> 8, dual perspective, SCReLU, QA=255/QB=64,
// 8 material-count output buckets (bucket = (popcount(occ)-2)/4, bullet's
// MaterialCount<8>) — adapted from bullet's examples/progression/
// 2_output_buckets.rs (pinned bullet rev in Cargo.toml). Data: bulletformat
// produced by `./chess datagen`.
//
// Usage: cargo run -r --bin trainer --features cuda -- <data.bin> [superbatches] [net_id]
//   shakedown: ~10 superbatches; full run: ~40.

use bullet_lib::{
    game::{inputs::Chess768, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

const HIDDEN_SIZE: usize = 512;
const OUTPUT_BUCKETS: usize = 8;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

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
        .unwrap_or_else(|| "hydray-ob-shakedown".to_string());

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        // Quantised layout consumed by sanity.rs and the engine loader:
        // l0w (768xH col-major, QA) | l0b (H, QA) |
        // l1w (OBx2H, QB, TRANSPOSED: each bucket's 2H weights contiguous,
        // for fast CPU inference) | l1b (OB, QA*QB)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104, // ~100M samples per superbatch
            start_superbatch: 1,
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
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data_path.as_str()]);

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

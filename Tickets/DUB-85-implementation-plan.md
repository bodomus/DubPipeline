# DUB-85 Implementation Plan

1. Extend `SourceSeparationConfig` and `SourceSeparationRequest` with native provider
   settings:
   - `model`;
   - `device`;
   - optional `model_file_dir`;
   - optional `output_format`;
   - optional `sample_rate`.
2. Add `AudioSeparatorProvider` in `dubpipeline/source_separation.py`:
   - import `audio_separator.separator.Separator` lazily;
   - resolve device from config and CUDA availability;
   - load the model lazily on first `separate()`;
   - reuse the same provider instance for subsequent requests;
   - call `Separator.separate()` and map native stems into DUB-84 output paths;
   - expose `close()` to release references and CUDA cache.
3. Keep `BsRoformerProvider` command-template behavior unchanged.
4. Update provider factory and metadata identity so `provider`, `model`,
   `model_path`, `model_file_dir`, device, and relevant native parameters invalidate
   stale cache.
5. Update step/CLI lifecycle:
   - let `step_source_separation.run()` accept an optional provider;
   - let `run_source_separation()` accept an optional provider instance;
   - create one source-separation provider in CLI `main()` for batch runs when the
     native provider is enabled;
   - close it in `finally` after all files.
6. Add focused unit tests using fake Separator classes:
   - provider selection;
   - no load on construction;
   - one `load_model()` across multiple separations;
   - cache hit skips native provider execution;
   - stable stem mapping by filename content, not output ordering;
   - import/load/separation/missing-stem/CUDA errors with strict and fallback modes.
7. Update `requirements.txt` and README only with minimal native-provider docs.
8. Run validation:
   - `python -m pytest tests/test_source_separation.py tests/test_step_merge_hq.py`;
   - `python -m pytest tests/test_cli.py`;
   - `python -m dubpipeline.cli --help`;
   - plan-mode run using a small temporary/native-provider pipeline.
9. Update CRG after changes and inspect impact with UTF-8 output settings if needed.
10. Attempt real manual validation only after dependency setup succeeds; otherwise
    document the blocked runtime validation and exact reason.
11. Create `review-DUB-85.md` with implementation report and limitations.

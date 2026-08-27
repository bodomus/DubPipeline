# DUB-84: Source Separation Background Provider

Source: https://bodomus.youtrack.cloud/issue/DUB-84

## Summary

Add a source-separation based background extraction path without replacing the existing ducking flow.

## Scope

- Add an audio background/provider boundary.
- Preserve current legacy ducking as the default behavior.
- Add an explicit `separated_background` mode.
- Support a BS Roformer provider with configurable model path and command, without hardcoded local paths or dependency auto-download.
- Persist stems under the pipeline workspace.
- Cache separation results by source audio identity and provider/model/material parameters.
- Handle failures explicitly, with legacy fallback only when configured.
- Integrate the step after audio extraction and before mix/mux.
- Add config, fake-provider/cache tests, and merge selection tests.

## Out of Scope

- DUB-85 residual vocal suppression and tuning.
- Mux redesign.
- Dependency/model auto-download.
- Replacing existing ducking behavior.

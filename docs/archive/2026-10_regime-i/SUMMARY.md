# Regime I campaign — 2026-10-07

ChatGPT work-mode completed the SmolLM2-135M reliability campaign (12 gates)
on an isolated shadow checkout. That package is now the canonical Gesso tree.

What was studied: sequential CPU/Lava/CUDA inference, loading, memory, numerics,
context, scheduling, containment, observability, a measured performance floor,
and a relocated fresh-depot packaging gate.

What was decided: Lava/Vulkan is the primary device path; CUDA is a throughput
comparator. The campaign stopped before Regime II. Palette and Cyan were not
edited.

What replaced it: live package files under `src/`, `ext/`, `test/`, and
`scripts/`, plus `docs/PACKAGING_REGIME_I.md` and
`docs/goals/REGIME_I_MATURITY.md`. Transfer blobs (patch, source tarball,
nested checkout, depots) stay in the gitignored drop folders
`work(from chatgpt)/` and `outputs(from chatgpt)/`.

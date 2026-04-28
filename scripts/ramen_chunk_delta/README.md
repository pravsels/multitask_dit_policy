# Ramen Chunk-Delta Investigation Scripts

Scripts that reproduce and diagnose the chunk-relative-vs-single-frame-delta
normalization bottleneck on the `villekuosmanen/bin_pick_pack_coffee_capsules`
dataset and the `pravsels/dit_coffee_capsules_config_fix` checkpoint.

See the implementation plan at
`docs/plans/2026-04-17-coffee-capsules-ramen-chunk-delta-fix.md` and the
written-up findings at
`docs/2026-04-17-coffee-capsules-ramen-investigation.md`.

Background analysis on the original block-tower checkpoint:
`docs/2026-04-16-ramen-chunk-delta-bottleneck.md`.

Run scripts in order (`01_*` → `06_*`); each writes outputs under
`scripts/ramen_chunk_delta/out/`.

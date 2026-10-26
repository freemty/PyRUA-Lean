# Reproduction notes: PyRUA-Lean Coder v0.1-4B on ORA's sweep-130

Run on fs-mbz-gpu-062 cards 4-7, 2026-10-25/26 (UTC). Outputs:
`/mnt/weka/shrd/hao/yuanbo/b2/outputs/pyrua-lean-sweep130/{smoke,full}/<cell>/`.

## Stack

- PyRUA-Lean: this fork, branch `freemty/sweep-130-repro` (upstream `ffad6d4` plus a host-side `ledger_span` per
  turn in `arms.py`; nothing the model sees changes).
- RPent `886b3b2` (the commit PyRUA-Lean's docs pin), venv `b2/envs/pyrua-lean-venv` (torch 2.7.1+cu128,
  curobo 0.7.8, rpent-liberopro 0.2.0, rlinf-robotwin-runtime 0.1.1, rlinf-lingbotvla 0.1.1).
- Weights (`~/yuanbo/models/pyrua-lean/MANIFEST.txt`): Coder 4B `0f80141`, Pi0.5 LIBERO `6222623`,
  LingBot-VLA RoboTwin `e727b46`; SAM3 `sam3.pt` (sha256 matches facebook/sam3).
- Settings of the paper's 4B evaluation (`docs/models/coder.md`): openai runtime, 40 cells, 5,400 s, 8,192
  output tokens, temperature 1.0, top_p 0.95, top_k 20, images on demand, pure feedback, RPent's guides, full
  primitive set (VLA on). Each episode boots its own simulator, VLA and (LIBERO) SAM3 servers.

## Deviations from the paper's setup

1. **Serving**: vLLM 0.32 (`repro/serve-coder.sh`), not SGLang; same `qwen3_coder` tool parser and `qwen3`
   reasoning parser, 65,536 context; vLLM has no `--constrained-json-disable-any-whitespace`.
2. **LIBERO rendering**: Mesa llvmpipe through EGL (`repro/launch.sh`), not NVIDIA EGL, so that several LIBERO
   simulators share a card and RoboTwin's SAPIEN runs on the same node.
3. **RPent `env.reset` timeout** raised from 120 s to 600 s in our RPent clone
   (`rpent/robots/components/env_client_base.py`) after RoboTwin resets timed out under load (and click_bell
   three times). It changes only how long the client waits.
   The RoboTwin runtime has a second 120 s limit on the server side (`robotwin/envs/vector_env.py`,
   `future.result(timeout=120)` around `setup_demo`, which raises an empty `TimeoutError`); raised to 600 s in our
   venv on 2026-10-26 04:55 after place_burger_fries, place_mouse_pad and place_dual_shoes hit it (original kept as
   `b2/pyrua-lean/setup/vector_env.py.orig`).
4. **Boot errors are rerun**: an episode whose host failed before the agent started (`status: boot_error`) is the
   infrastructure's and is run again (`run_sweep.py`); every other outcome counts as the cell's one attempt.

## Cells

- 130 cells of `repro/sweep-130.json` (LIBERO-Pro 80: eight suites x tasks 0-9, initial state 1; RoboTwin 50).
- RPent's LIBERO seed is the initial-state index (`env_server.make_env`: `seed % len(init_states)`), so ORA's
  `seed: 1` maps to `--seed 1`.
- **Not runnable on RPent's LIBERO-PRO (3)**: `libero_spatial_task` t3 and t7, `libero_10_task` t2 have zero
  initial states in rpent-liberopro 0.2.0 (364-byte `.pruned_init` files); ORA's LIBERO-Pro data has them.
- **Scene differs from ORA's (17)**: the other `libero_spatial_task` and `libero_10_task` cells. RPent's package
  re-pruned every init file of these two suites (all 10 differ from ORA's) and 3 of their BDDL files differ, so
  initial state 1 is not ORA's scene. The other six suites (60 cells) are byte-identical in BDDL and init files.
- RoboTwin: same task names and seeds as ORA (`demo_randomized`); the asset root uses RoboTwin2.0's pinned
  `embodiments.zip` (c15cc97) with background textures and objects from the existing tree (identical by LFS hash).

# Placeholder for downloaded / exported policy checkpoints.
#
# Phase 0: no checkpoints stored yet.
# Phase 2: Stage A source policy or demonstration bundle
# Phase 7: Stage C fine-tuned adapter weights
#
# Expected layout:
#   checkpoints/
#     stage_A/          # source policy or demo index
#     stage_B/          # retargeting-only (deterministic, may be empty)
#     stage_C/          # fine-tuned residual adapter

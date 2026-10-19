0051a — Puffer minimal staging YAML-forwarding hotfix
=====================================================
Prerequisite: 0051 (after 0050a); do not apply directly to 0050a.

BUG:
0051 added sdf_puffer_minimal_staging: true to
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0050.yaml,
but action_part_adapt omitted it from the top-level YAML allowlist.
As a result, the value was dropped when loading extraction_config, and
puffer still wrote the full 5.815 GB temporary localization payload.

FIX:
Add 'sdf_puffer_minimal_staging' to the top-level extraction-config
allowlist in src/app/agent_3dvec_sdf.py. No SDF/localization formula changes.
Add a regression to ensure the YAML opt-in stays reachable.

INSTALL after 0051:
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0051a-puffer-config-forwarding-files.zip

Alternatively apply the Git patch (not both ZIP and patch):
  git apply --check ~/Downloads/oktopus-0051a-puffer-config-forwarding.patch
  git apply ~/Downloads/oktopus-0051a-puffer-config-forwarding.patch

Run the same puffer 512 command and YAML as before.
Expected early log:
  [sdf_puffer_0051] name= puffer_body_wrap_no_snug minimal_correction_staging=True
And after localization:
  [sdf_puffer_0051] minimal_staging=True ... inside_staged=False

If the job is already running, it will continue with its loaded Python
code. Restart the job to activate the source change.

Test:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_puffer_minimal_config_0051a.py' -v

Local validation: 2 new forwarding checks, 11 prior puffer staged-field parity
and wrap guard checks, plus Python compilation passed. No live puffer
runtime measured for 0051a yet. Do not run 2048 until 512 parity confirmed.

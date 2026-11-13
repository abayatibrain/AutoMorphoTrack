AutoMorphoTrack 2.4.0 guarded integration

Install:
python -m pip install release_artifacts/2.4.0/automorphotrack-2.4.0-py3-none-any.whl

Check installed core and known-motion outputs:
python -I validation/check_installed_core.py

Guarded fixture:
python -I integration/run_guarded.py --input validation/known_motion.ome.tif --output ../amt-guarded-check --config validation/known_motion_config.json --wheel release_artifacts/2.4.0/automorphotrack-2.4.0-py3-none-any.whl

Other environments, explicitly recorded as outside the retained environment profile:
python -I integration/run_guarded.py --input validation/known_motion.ome.tif --output ../amt-other-environment --config validation/known_motion_config.json --wheel release_artifacts/2.4.0/automorphotrack-2.4.0-py3-none-any.whl --allow-unvalidated-environment

Python API: automorphotrack.validated
CLI: amt-validated
Internal core revision: 1.1.0rc2
Public distribution: 2.4.0

The isolated guarded process aliases the checked core to automorphotrack internally. This preserves the original adapter contract. The normal installed API is automorphotrack.validated.

Authoring contract:
- Keep the 18 core module bytes unchanged.
- Put orchestration, configuration and reporting in an external adapter.
- Record input hashes, true axes, channel mapping, settings and preprocessing.
- Do not infer physical calibration or biological replication from a file name.
- Compare the fixture outputs before interpreting new data.
- An unchanged core does not establish accuracy on a new image domain.
- The guard checks consistency and is not a security sandbox.

Gemini adapter origin: Gemini app output, locally executed by Codex.
The app did not execute the local analysis itself.
Adapter SHA256: 902b4832e27c6a204aaf755ddaf741ac983e6a3b6ec2f81bb7f2e21f0c679145

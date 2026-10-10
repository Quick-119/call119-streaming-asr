# Public release boundary

The real 119-call recordings, call-level reference transcripts, IDs, manifests, model outputs, and training windows stay outside this repository. The real-call study here contains aggregate CER and sample counts only. The WAV files under `benchmark/mock119_30/audio/` are **newly generated synthetic conversations** from the published fictional scripts; no real-call audio was mixed into them.

The model checkpoint, adapters, VAD weight file, and complete frozen runtime bundles are also excluded. Users receive them separately through an access-controlled channel if authorized. This avoids putting multi-gigabyte checkpoints or data-bearing research directories into Git history.

For future updates, copy only reviewed files into this repository. `.gitignore` blocks common dataset, transcript, output, and checkpoint paths. `scripts/audit_public_tree.py` checks every tracked path against a narrow allowlist, rejects credentials and workspace paths in text files, limits file sizes, and verifies the synthetic WAV hashes against the benchmark manifest. Run it before every push. The audit is a release gate; it does not authorize adding source data.

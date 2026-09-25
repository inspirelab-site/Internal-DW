# Public release checklist

## Required before the first GitHub release

- [ ] Select and add an open-source `LICENSE` with all authors' approval.
- [ ] Replace anonymous paper metadata with author and citation information.
- [ ] Attach a versioned result/checkpoint bundle and publish SHA-256 checksums.
- [ ] Document the upstream URLs and licenses for HCP, The Well, ETT, and
      WeatherBench-2; do not redistribute restricted raw data.
- [ ] Run `bash scripts/reproduce/00_smoke_test.sh` in a fresh Linux environment.
- [ ] Run `python -m pytest -q`.
- [ ] Run every `render` entry point against the release result bundle.
- [ ] Inspect `git ls-files` and `git diff --cached`: tracked/staged files
      must contain no credentials, machine-specific paths, raw participant
      data, checkpoints, logs, or cluster launch scripts.

## Suggested release contents

- Git repository: source, tests, documentation, examples, and the explicitly
  allowlisted paper scripts.
- Release asset `internal-dw-results-<version>.tar.zst`: compact JSON/NPZ
  ledgers and, where redistribution permits, trained checkpoints.
- `SHA256SUMS`: checksum for every release asset.

Do not force-add ignored operational scripts or generated outputs.

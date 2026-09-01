# Publishing this repository

The research checkout contains private data, checkpoints, cluster logs, and
discarded experiments. The root `.gitignore` is therefore an explicit public
allowlist for `scripts/`; do not weaken it merely to make a local file appear
on GitHub.

## 1. Decide the public metadata

Before publishing, add the intended `LICENSE` and update the citation section
of `README.md` (and optionally add `CITATION.cff`). If the paper is under
anonymous review, verify the venue's current anonymity policy before creating
a public repository tied to an author account.

## 2. Inspect the exact first commit

From the repository root:

```bash
git init
git branch -M main
git add --dry-run .
git add .
git status --short
git diff --cached --stat
git diff --cached
```

The staged tree should contain source, tests, documentation, examples, and the
paper reproduction scripts. It should not contain `data/`, `artifacts/`,
`experiments/`, `probe_outputs/`, `logs/`, checkpoints, generated figures, or
root-level research ledgers.

Check for unexpectedly large staged files before committing:

```bash
git ls-files -z | xargs -0 du -h | sort -h | tail -n 30
```

On PowerShell, the corresponding check is:

```powershell
git ls-files | ForEach-Object { Get-Item -LiteralPath $_ } |
  Sort-Object Length -Descending |
  Select-Object -First 30 FullName, Length
```

Also inspect the staged diff for host paths, usernames, tokens, private keys,
and service credentials. Removing a secret in a later commit does not remove
it from Git history.

## 3. Make and push the initial commit

Create an empty GitHub repository without an auto-generated README or license,
then run either the SSH form:

```bash
git commit -m "Initial public release"
git remote add origin git@github.com:OWNER/REPOSITORY.git
git push -u origin main
```

or the HTTPS form:

```bash
git commit -m "Initial public release"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

If GitHub CLI is installed, repository creation and the first push can instead
be combined with:

```bash
gh repo create OWNER/REPOSITORY --public --source=. --remote=origin --push
```

Run `git remote -v` before the first push to catch an incorrect owner or
repository name.

## 4. Publish result artifacts separately

Large checkpoints, licensed datasets, and generated experiment trees are not
Git source. Publish only redistributable checkpoints and compact result ledgers
as a versioned artifact bundle on an archival service or a GitHub release, and
record its URL and SHA-256 checksum in `README.md`. Preserve the relative paths
expected by `scripts/reproduce/` when constructing the archive.

Do not redistribute HCP, The Well, WeatherBench-2, or any other upstream data
unless its license explicitly permits it. Prefer download/preparation
instructions over copying datasets into the repository. Git LFS is useful only
for versioned files that genuinely belong in Git; it is not a substitute for a
dataset or checkpoint archive.

## 5. Tag a reproducible release

After the source commit and artifact bundle are final:

```bash
git tag -a v0.1.0 -m "Paper release v0.1.0"
git push origin v0.1.0
```

Create a GitHub release from that tag, attach or link the matching result
bundle, and include the source commit, artifact checksum, environment notes,
and any known reproduction caveats in the release notes.


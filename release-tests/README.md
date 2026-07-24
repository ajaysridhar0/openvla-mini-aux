# BARX release-test evidence

Independent runners keep raw command evidence outside the Git checkout so a
test cannot make the repository dirty. Use one immutable directory per
repository commit, protocol revision, runner, command, and retry:

```text
runs/<full-commit>/<protocol-hash>/<runner-id>/
  run.json
  commands.jsonl
  commands/S08-C03-A01.sh
  logs/S08-C03-A01.stdout.txt
  logs/S08-C03-A01.stderr.txt
  artifacts.jsonl
  visual-review.md
  report.md
```

`scripts/log_release_command.py` copies a literal shell script into that
directory, runs it, mirrors its output, and records timestamps, duration, exit
code, and SHA-256 hashes. Example:

```bash
python scripts/log_release_command.py \
  --run-dir "$BARX_RUN_EVIDENCE" \
  --command-id S08-C03 --attempt 1 --section mimicgen \
  --cwd "$BARX_REPO" --script "$BARX_COMMAND_SCRIPTS/S08-C03.sh"
```

Never overwrite a failed attempt. Increment `--attempt`, classify the retry in
`report.md`, and say whether it used clean state. A changed repository commit
always starts a new run. The release receives an overall pass only when every
required section passes at one exact commit.

Raw logs may contain machine paths and remain outside the public repository.
The root `RELEASE_TEST_COMMAND_LOG.md` and `RELEASE_ACCEPTANCE_REPORT.md` are
the sanitized, reviewable summaries: replace machine roots with `$BARX_*`,
omit credentials and private paths, retain every literal command and exit
code, and link each conclusion to its run, artifact checksum, or visual review.

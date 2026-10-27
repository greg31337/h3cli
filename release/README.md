# Release commands

Use the [release guide](../docs/release/README.md). The public interface is:

```sh
./release/release.sh setup-signing
./release/release.sh configure --version v0.6.0
./release/release.sh build-local
./release/release.sh test-local
./release/release.sh build-packages
./release/release.sh test-packages
./release/release.sh publish
```

`flow.py` coordinates the stages, `worker.py` performs native host work,
`native.py` builds/tests recorded native artifacts, and `common.py` shares
logging, cancellation and receipt checks. Configuration and all evidence are
ignored under `outputs/release/`. `--help` describes each command; `--dry-run`
prints a plan without executing work.

The numbered platform scripts and the old driver/coordinator are legacy
interfaces retained for historical diagnostics and tests. They are not needed
for a new release. Do not mix their manual run IDs or receipts with this flow.

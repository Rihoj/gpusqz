# Releases and versioning

gpusqz follows [semantic versioning](https://semver.org) and is released
automatically by [semantic-release](https://semantic-release.gitbook.io)
from the commit messages on `main`, which follow [Conventional
Commits](https://www.conventionalcommits.org):

| commit | example | release |
|---|---|---|
| `fix:` / `perf:` | `fix(vulkan): retry refused allocations` | patch (0.1.0 → 0.1.1) |
| `feat:` | `feat: add --level` | minor (0.1.0 → 0.2.0) |
| breaking: `!` after the type, or a `BREAKING CHANGE:` footer | `feat!: format v2` | minor while on 0.x; major from 1.0 on |
| anything else (`docs:`, `test:`, `ci:`, `chore:`, `refactor:`, …) | `docs: fix typo` | none |

While the version is 0.x, the `.gsz` format and the command line may
still change between minor versions. Going to 1.0.0 is a deliberate step:
remove the `"breaking": true → minor` rule from `.releaserc.json`, then
merge a breaking change.

## What CI does

On every push to `main` the `build` workflow asks semantic-release for the
next version (dry run), builds and tests every package with it, and only
when all of them pass tags the commit `vX.Y.Z` and publishes the GitHub
release with generated notes and the packages attached. A push with
nothing releasable just builds. Pull request titles are checked against
Conventional Commits, because a squash merge turns the title into the
commit message; use squash merges (or write every commit that way).

Never create or push `v*` tags by hand, and never edit a version number
in the source: the tag and the packages' version both come from CI.

## Version strings

`gpusqz --version` and `gpusqz_refdec --version` print the version. A
release build prints `X.Y.Z`; any other build prints `git describe`'s
view, e.g. `0.1.0-3-gabc1234` (3 commits after v0.1.0) with `-dirty` for
uncommitted changes, fixed when CMake configures. Configure with
`-DGPUSQZ_VERSION=X.Y.Z` to set it explicitly. The `v0.0.0` tag is not a
release: it marks where the commit history starts to count.

The release tooling is pinned in `release/package.json` (with its
lockfile); `release/next-version.mjs` is the dry run CI uses.

## Performance and releases

A release that changes speed or ratio should have a matching entry in
[Performance history](performance-history.md), with the version in its
release column.

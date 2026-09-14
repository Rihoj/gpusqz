---
name: ship
description: Take a gpusqz change from a branch to a published release - Conventional Commit messages, PR with a checked title, watching and debugging the 4-platform CI, merging, verifying the GitHub release and its 6 packages, and mirroring to the Gitea remote. Use when the user asks to open a PR, merge, release, or check CI.
---

# Ship a change

Releases are automatic: every push to `main` runs `.github/workflows/build.yml`,
and semantic-release publishes a version if the new commits call for one.
Your job is to get the commit messages right and CI green.

## 1. Commits

Conventional Commits, because they choose the version:

| type | release (0.x) |
|---|---|
| `fix:` `perf:` | patch |
| `feat:` | minor |
| `feat!:` / `BREAKING CHANGE:` footer | minor (major from 1.0 on) |
| `docs:` `test:` `ci:` `build:` `chore:` `refactor:` `style:` | none |

Add a scope where it helps (`fix(vulkan):`, `feat(cli):`). Say in the body
what changed for users; the release notes are generated from these
subjects. Never hand-edit versions or create `v*` tags.

A `perf:` commit, or any change that moves speed or ratio, carries its
measurements into `docs/performance-history.md` in the same PR (see the
`/benchmark` skill); add the release version to that entry once it ships.

## 2. PR

```
git push github HEAD:refs/heads/<branch>
gh pr create -R Rihoj/gpusqz --base main --head <branch> --title "<conventional title>" --body "..."
```

The `pr-title` job checks the title (a squash merge uses it as the commit
message). A merge commit keeps each commit's own message for the notes.

## 3. CI

```
gh run list -R Rihoj/gpusqz --branch <branch> --limit 1
gh run watch <id> -R Rihoj/gpusqz --interval 30 > /dev/null; gh run view <id> -R Rihoj/gpusqz
gh run view -R Rihoj/gpusqz --job <job-id> --log > job.log      # then grep it
```

Jobs: `version`, `pr-title`, Linux deb (runs the Vulkan suite on lavapipe),
Linux rpm (Rocky 8 container; sometimes slow downloads), Windows (zip + msi,
installs/uninstalls the msi), macOS (tar.gz + pkg, installs the pkg).
Known traps are in CLAUDE.md (PowerShell `-D` quoting, the macOS runner's
unusable GPU, CRLF fixtures, CPack productbuild components). CUDA kernels
can't run in CI: run `ctest --test-dir build` locally on the GPU first.

To try a workflow change without a PR: push a branch and
`gh workflow run build.yml -R Rihoj/gpusqz --ref <branch>`.

## 4. Merge and verify the release

```
gh pr merge <n> -R Rihoj/gpusqz --merge
gh run list -R Rihoj/gpusqz --branch main --limit 1      # watch it as above
gh release view -R Rihoj/gpusqz --json tagName,assets --jq '.tagName, .assets[].name'
```

A release must have 6 assets, all named with its version: `.deb`, `.rpm`,
`-win64.msi`, `-win64.zip`, `-Darwin.pkg`, `-Darwin.tar.gz`. If the release
job refuses ("next version is now ..."), another release landed during
the build: re-run the workflow on `main`.

## 5. Mirror to Gitea

`origin` is the user's self-hosted Gitea (ssh, port 2222). Its key has a
passphrase, so pushing only works when the user's ssh-agent holds it:
`git push origin main --follow-tags`. If it's refused, tell the user rather
than retrying.

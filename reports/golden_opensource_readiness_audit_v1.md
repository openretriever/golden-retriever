# GoldenRetriever Open-Source Readiness Audit v1

Scope: read-only audit of GoldenRetriever for public-release readiness. This repo is separate from the core `retriever` release candidate.

## Live State

- Current release-prep branch: `chore/remove-golden-src-20260621`.
- This branch removes the legacy `src/golden_retriever` package and treats Golden as an examples/docs companion to the core `retriever` runtime.
- Local untracked planning/report artifacts may still exist under `reports/`; they are not part of the public tree unless explicitly added.
- No remote fetch was performed during the original audit; refresh status before any public push.

## Release Blockers

1. Root release metadata needs final verification.
   - Root `pyproject.toml`, `LICENSE`, `CONTRIBUTING.md`, and `SECURITY.md` now exist for the companion examples repo.
   - A `NOTICE` file and GitHub issue/PR templates can be added when the public repository is created.
   - Package metadata for nested or optional packages is still not intended as a public distribution boundary.

2. Public onboarding depends on sibling development checkouts.
   - Root README and Pixi config now use the temporary `debug-retriever` package instead of a sibling checkout.
   - This should switch to the public core `retriever` package once it is published.
   - Public setup should continue to work from the repo itself, with editable-core guidance kept out of the default path.

3. Machine-specific paths remain in active code/config.
   - Active source contains `/scratch/...` and `/home/...` defaults in segmentation/model server, VLMaps, D3Fields, robomimic, and legacy robot/skill code.
   - These must be parameterized, moved behind archive boundaries, or removed before a public cut.

4. Source tree mixes code, generated outputs, model assets, and vendored payloads.
   - Tracked heavyweight files include VLF-M model weights, Habitat assets, kitchen/CLIport meshes/textures, tokenizer payloads, generated wheel artifacts, notebooks, HTML/PDF outputs, and benchmark assets.
   - Decide what ships in git, what moves to release artifacts or LFS, and what should be excluded entirely.

5. Dependencies remain developer-oriented.
   - Direct Git dependencies and local editable-core assumptions make the public install path fragile.
   - Public package metadata should use stable dependencies or clearly documented optional extras.

6. Public docs still need a final staging-vs-public sweep.
   - Archive material and historical carry-back notes should remain clearly separated from the public quickstart.
   - The front door should keep concise example lanes first and push heavier prototypes to secondary docs.

## Recommended Cleanup Order

1. Keep Golden public scope as a public examples/docs companion repo, not the core runtime.
2. Continue using a dedicated release-prep branch rather than publishing an old staging `main`.
3. Add final GitHub issue/PR templates and optional `NOTICE` policy when the public repo is created.
4. Split current tree into keep/archive/artifact/drop buckets.
5. Remove or parameterize hardcoded local paths in active code.
6. Move heavyweight assets and generated outputs out of git unless intentionally shipped with documented provenance.
7. Rewrite public onboarding so default setup does not depend on sibling local repos.
8. Run a private-content scan, license scan, and clean-clone install/test pass before any public remote push.

## Relationship To Core Retriever Release

Golden should not block the core `retriever` release candidate. The core release should proceed in the separate release repo using the filtered history candidate. Golden can later consume the public core package once its own release boundary is clean.

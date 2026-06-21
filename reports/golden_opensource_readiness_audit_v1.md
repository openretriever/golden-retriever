# GoldenRetriever Open-Source Readiness Audit v1

Scope: read-only audit of GoldenRetriever for public-release readiness. This repo is separate from the core `retriever` release candidate.

## Live State

- Current branch status: `main...origin/main [ahead 21, behind 1]` with untracked `reports/`.
- Treat local `main` as a staging stack, not a clean public release branch.
- No remote fetch was performed during this audit.

## Release Blockers

1. Root release metadata is not ready.
   - Root `pyproject.toml` is a placeholder/future template and declares proprietary license text.
   - No root `LICENSE`, `NOTICE`, `CONTRIBUTING.md`, `SECURITY.md`, or `.github/` release surface was found.
   - Package metadata for nested packages is incomplete for public distribution.

2. Public onboarding depends on sibling development checkouts.
   - Root README and pixi config still describe resolving `retriever` from a sibling `retriever-mirror` checkout.
   - Several example docs present local editable-core setup as part of the active path.
   - Public setup should work from the repo itself, with editable-core guidance moved to a contributor-only section.

3. Machine-specific paths remain in active code/config.
   - Active source contains `/scratch/...` and `/home/...` defaults in segmentation/model server, VLMaps, D3Fields, robomimic, and legacy robot/skill code.
   - These must be parameterized, moved behind archive boundaries, or removed before a public cut.

4. Source tree mixes code, generated outputs, model assets, and vendored payloads.
   - Tracked heavyweight files include VLF-M model weights, Habitat assets, kitchen/CLIport meshes/textures, tokenizer payloads, generated wheel artifacts, notebooks, HTML/PDF outputs, and benchmark assets.
   - Decide what ships in git, what moves to release artifacts or LFS, and what should be excluded entirely.

5. Dependencies remain developer-oriented.
   - Direct Git dependencies and local editable-core assumptions make the public install path fragile.
   - Public package metadata should use stable dependencies or clearly documented optional extras.

6. Public docs still include internal-development framing.
   - Local editable-core docs, mirror carry-back notes, advanced examples, and archive material should be clearly separated from the public quickstart.

## Recommended Cleanup Order

1. Freeze Golden public scope: decide whether this repo is public examples, research artifacts, or a private staging stack.
2. Create a dedicated release branch instead of using the current ahead/behind `main`.
3. Add root OSS metadata: final license, notice policy, contributing/security stubs, package URLs, and ownership docs.
4. Split current tree into keep/archive/artifact/drop buckets.
5. Remove or parameterize hardcoded local paths in active code.
6. Move heavyweight assets and generated outputs out of git unless intentionally shipped with documented provenance.
7. Rewrite public onboarding so default setup does not depend on sibling local repos.
8. Run a private-content scan, license scan, and clean-clone install/test pass before any public remote push.

## Relationship To Core Retriever Release

Golden should not block the core `retriever` release candidate. The core release should proceed in the separate release repo using the filtered history candidate. Golden can later consume the public core package once its own release boundary is clean.

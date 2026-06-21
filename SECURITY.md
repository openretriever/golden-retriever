# Security Policy

GoldenRetriever is early-stage research and examples software. Do not use it as a security boundary for robot operation.

## Reporting

For now, report security issues privately through the repository owner or the OpenRetriever maintainers before opening a public issue.

Include:

- affected files or examples;
- reproduction steps;
- whether the issue involves credentials, local filesystem exposure, remote code execution, unsafe robot control, or dependency supply-chain risk.

## Secrets And Local State

Do not commit:

- API keys, tokens, certificates, robot credentials, or private endpoints;
- local absolute paths;
- logs or recordings that contain private environment details;
- unpublished model weights, datasets, or papers.

Prefer environment variables and `.env` files ignored by Git for local credentials.

---
description: Scan for security issues (deps, secrets, common pitfalls). Apply security-scanner subagent.
---

# /security

## Procedure

1. **Secret scan.** Run `gitleaks detect --no-banner --redact` on the working tree. Zero hits is the bar.
2. **Dependency audit.**
   - Python: `uv run pip-audit` (or `uv pip audit` if available). Surface critical/high CVEs first.
   - Node tooling: `npm audit --omit=dev` if `package.json` exists.
3. **Code review pass.** Delegate to `security-scanner` subagent. Focus on:
   - Secrets in code/logs (hardcoded tokens, debug prints of env, secrets in error messages).
   - Input validation at trust boundaries (CLI flags, HTTP, file paths from user).
   - Command injection (`subprocess` with `shell=True`, untrusted strings into shell).
   - Path traversal (joining user-supplied paths without validation).
   - SQL injection (string concat into queries).
   - Unsafe deserialization (`pickle`, `yaml.load` without `safe_load`).
   - Crypto: hand-rolled crypto, weak algorithms, missing constant-time compare.
   - Permissions: secret files should be `0600`; env files should be `.gitignore`-d.
4. **Report** in three buckets: 🚨 fix now, ⚠️ fix soon, 💡 hardening ideas.

## Don't

- Don't auto-fix CVEs by bumping deps blindly — read the changelog for breaking changes first.
- Don't suggest installing unverified third-party scanners; stick to the ones in the meta-repo's CI baseline.

# Engineering rules

- Keep this a small local Python utility for controlled Open WebUI sampling and personality experiments. Open WebUI and Ollama perform inference; this project automates experiments and reporting.
- Prefer the Python standard library. Do not add unnecessary frameworks, GUI, database, web server, automated scoring, or architecture.
- The selected Open WebUI preset supplies the baseline. Override only parameters and seeds actively controlled by the experiment; leave unspecified settings to Open WebUI/backend defaults.
- Every sample uses an independent fresh user-only context unless a different context design is explicitly requested. Never copy a preset's system prompt into this project.
- Never expose API credentials. Keep `.env`, generated results, caches, and local artifacts out of public commits. Use project-relative resource paths.
- Preserve the shared report theme, actual parameter/seed recording, and reproducibility metadata. Do not imply fixed seeds guarantee determinism across backends or versions.
- Make targeted changes only. Working behavior is finished unless there is a concrete problem or a requested feature.
- Run syntax validation and relevant offline checks after changes. Do not run model generations unless explicitly requested.

# Validated core records

The public distribution packages two explicitly separated interfaces. The manuscript uses `automorphotrack.validated`; the earlier interface remains at `automorphotrack`.

| Route | Entry point | Evidence scope |
| --- | --- | --- |
| Manuscript Python API | `automorphotrack.validated.run_pipeline` | Frozen core and recorded validation |
| Manuscript CLI | `amt-validated` | Same frozen core |
| Existing Python API | `automorphotrack` | Earlier development interface |
| Existing GUI and MCP | `automorphotrack gui`, `automorphotrack mcp` | Earlier development interface |

The internal core revision identifier remains `1.1.0rc2` because the quantitative modules are unchanged. Public distribution version `2.4.0` is additive to repository version `2.3.0` and follows PyPI version `2.1.0`. It does not retroactively change earlier manifests or claim that older interfaces passed the new benchmark.

The archived 1.1.0rc2 wheel and its original integration receipts remain the exact historical reproduction artifacts. Release verification compares the installed public namespace with those retained outputs. CLI, notebook, and Gemini-authored adapter results are execution checks, separate from biological validation.

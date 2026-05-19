# AutoMorphoTrack MCP Connector

AutoMorphoTrack ships an [MCP](https://modelcontextprotocol.io) server so any
MCP-aware AI client — **Claude Code**, **Claude Desktop**, **Cursor**, etc. —
can drive the pipeline from natural-language prompts. This is the concrete,
implemented "AI-assisted" interface that Reviewers #1, #2, and #3 asked for.

## Install

```bash
pip install "automorphotrack[mcp]"
```

The `[mcp]` extra adds the official `mcp` Python package; the core
AutoMorphoTrack install does not depend on it, so users who only want the
notebook / CLI workflow do not pay the dependency cost.

## Register with Claude Code

```bash
claude mcp add automorphotrack -- automorphotrack mcp
```

That tells Claude Code to launch `automorphotrack mcp` (a CLI sub-command)
whenever it needs the AutoMorphoTrack tools. From that point on you can say
things like:

> "Use automorphotrack to run the full pipeline on `./stack.tif`, channel 0
> for mitochondria and channel 1 for lysosomes. Then summarize the motility
> distribution and the Manders M1 coefficient."

Claude Code will:
1. Call `amt_run(tif_path="./stack.tif", mito_channel=0, lyso_channel=1)`.
2. Read the resulting CSV summaries via `amt_describe_outputs`.
3. Compose the answer from the actual files.

## Register with Claude Desktop

Add this block to your `claude_desktop_config.json` (`~/Library/Application
Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "automorphotrack": {
      "command": "automorphotrack",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Desktop; the AMT tools will appear in the tools panel.

## Tools exposed

| Tool | What it does |
|---|---|
| `amt_run` | Full pipeline on a TIF stack |
| `amt_detect` | Detection only |
| `amt_shape_features` | Per-organelle shape descriptors |
| `amt_motility` | Velocity / displacement from track CSVs |
| `amt_colocalization` | Manders / Jaccard / Pearson / cosine |
| `amt_summary` | Integrated Spearman correlation matrix |
| `amt_validate_segmentation` | Dice / IoU / precision / recall / F1 |
| `amt_sensitivity_analysis` | Parameter sweep with metrics per value |
| `amt_benchmark_export` | Export AMT output in CellProfiler / MiNA / MitoGraph schema |
| `amt_describe_outputs` | Walk an output folder and summarize its CSVs |
| `amt_list_capabilities` | Machine-readable capability map |

## Verifying it works

```bash
# In one terminal:
automorphotrack mcp

# In another (with @modelcontextprotocol/inspector):
npx @modelcontextprotocol/inspector automorphotrack mcp
```

The inspector UI will show every tool above and let you call them
interactively.

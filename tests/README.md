# Regression tests

Run the complete suite from the repository root:

```bash
python -m unittest discover -s tests/unit -v
```

The suite covers the corrected writer evidence contract, citation rendering,
citation checking, and result rescoring. It also checks the Anthropic and
OpenAI provider adapters, the Codex and Claude subscription adapters, and the
portable skill and plugin package.

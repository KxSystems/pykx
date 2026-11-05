# AGENTS.md

Guidance for AI agents working in this repository. `CLAUDE.md` is a symlink to this file, so Claude Code and other agents read the same instructions.

## About this repo

This repo is KDB-X Python (`pykx`), a Python interface to q. It has three parts:

- Code: the package is in `src/`. Cython is used only where it is needed. Do not edit the generated C files.
- Tests: in `tests/`, run with pytest. The settings are in `tests-config.yml` and `pyproject.toml`.
- Documentation: in `docs/`, built with MkDocs and the Material for MkDocs theme. The site is published at `code.kx.com/pykx`. The site structure is defined by the `nav:` section in `mkdocs.yml`.

The development guide is [`DEVELOPING.md`](DEVELOPING.md). Read it before you change the API, because it sets the rules for new public functions and methods.

## Core principles

- Verify product facts before you write them. Check the behaviour in the source in `src/` or in a q session. Do not write KDB-X Python behaviour from memory, or from another library.
- Find the canonical page before you add a new one. Link to it rather than repeating its content.
- Put each detail in one place. If two pages need the same detail, one of them links to the other.
- The primary readers of the package are Python users who may not know q. Write for them first, and explain q terms on first use.
- Existing pages do not always follow these rules. Do not copy a pattern from a page without checking it against this file.

## Writing and style

- Never hard-wrap prose. Write each paragraph and each list item as one long line, and let the editor or renderer wrap it.
- Use US English. The pull request template asks for spell checking against US English.
- Label code blocks with their language.
- Use Python conventions for code. [`DEVELOPING.md`](DEVELOPING.md) cites PEP 8, PEP 257 and PEP 484.

## Commands

```bash
pytest tests                                  # run the test suite; read tests/ and tests-config.yml for setup first
mkdocs serve                                  # live docs preview at http://localhost:8080 (dev_addr in mkdocs.yml)
```

Install the docs dependencies listed in `pyproject.toml` before you run `mkdocs serve`. `mkdocs` is pinned to a 1.5 release line, so use the pinned version.

## Before you open a PR

- [ ] The tests you changed or added pass.
- [ ] The docs build and the pages you changed render correctly in `mkdocs serve`.
- [ ] Each new documentation page is in `nav:` in `mkdocs.yml`. A new class also needs a documentation stub, as the documentation PR template asks.
- [ ] Links, including anchors, work.
- [ ] The PR uses the matching template in `.github/PULL_REQUEST_TEMPLATE/`.

## Boundaries

- Do not add credentials, tokens, customer data or internal hostnames or links to any file. This repo is public.
- Do not commit generated files, such as the Cython C output or the built site.
- Do not edit the generated C files. Fix the Cython source instead, as [`DEVELOPING.md`](DEVELOPING.md) describes.
- Do not open a PR, push, or change the public API unless the user asks.

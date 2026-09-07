# CLAUDE.md — abczarr

Guidance for coding agents working in this repository.

## Documentation style

This applies to every piece of prose that ships with the code: the README, the
documentation site (`docs/`), and every public docstring (anything without a
leading underscore, rendered into the API reference by mkdocstrings). All
documentation is written by Sonnet.

Write for the developer who will *use* the code — not its author, and not a
future maintainer of its internals. Every sentence earns its place by helping
that reader write correct, readable code against the public surface.

**Voice**

- Write for humans, and write nicely. "Technical" means precise and reasonably
  short; it does not mean terse, clunky, or ugly. Aim for clean, well-formed
  prose that happens to be exact.
- No agentic style, anywhere, at all. No "Let me…", no "we will now…", no
  meta-commentary about the writing, no hedging, no filler throat-clearing, none
  of the tells of machine-written text. It should read as if a careful engineer
  wrote it by hand.

**Content**

- Say what it does, not why or how it is implemented. Lead with the concept and
  what the caller gets back. Implementation detail — lazy imports, import
  cycles, caching, which private helper does the work, how a value is stored —
  never belongs in a docstring. If such rationale is worth recording, it goes in
  a code comment near the implementation.
- Be lean. Cut every word that can go without losing meaning. A one-line
  docstring is the right length when that is all it needs; do not inflate it
  into a numpydoc block for form's sake. Use Parameters / Returns / Raises only
  when there is something worth listing, and keep them tight.
- Do not narrate history or decisions. "deliberately", "we now", "rather than",
  "note that we chose" describe a changelog, not behaviour. State the behaviour;
  the history belongs in the commit message.
- Lead with the simplest spelling. Any runnable example (a `pycon` block) must
  be copy-pasteable, true to what the interpreter prints, and pass on the oldest
  supported Python (3.8).
- A public docstring is a rendered documentation page. It must not reference an
  issue or PR number or a review finding, name a private symbol the reader never
  touches, or read like a note to a fellow contributor.
- Match the tone of the repository's existing good public docstrings. Read a few
  before you start and write in that voice.

**Editing existing documentation**

Understand the core of what the text is trying to say, then write the best
version of it. Do not make timid, surface-level edits. If a docstring or
paragraph is muddled, agentic, or over-explained, rewrite it completely.

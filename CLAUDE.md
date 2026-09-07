# CLAUDE.md — abczarr

Guidance for coding agents working in this repository.

## Documentation style

This applies to every piece of prose that ships with the code: the README, the
documentation site (`docs/`), and every public docstring (any name without a
leading underscore, which mkdocstrings renders directly into the API
reference). All documentation is written by Sonnet.

Write for the developer who will use the code, not for its author and not for a
future maintainer of its internals. The goal is low cognitive load. A reader
should spend attention on the behaviour and purpose of the code, not on
decompressing dense sentences, resolving ambiguous pronouns, or filtering out
conversational filler.

**Sentences and clarity**

- Write complete, well-constructed sentences. Prefer simple, direct sentences
  to dense ones. Documentation may be short and still use full sentences. The
  aim is lean prose, not telegraphic prose.
- Keep to one clear logical relationship per sentence. When a sentence carries
  several distinct ideas, write several sentences. Do not join independent
  ideas with commas, semicolons, em dashes, or parenthetical clauses.
- Do not open with a fragment. A docstring that begins "Common base for…",
  "The mirror of…", or "One transform for…" asks the reader to supply the
  missing subject. Begin with an explicit subject and a verb: "This class is
  the base class for…", "An `Affine` transform maps…".
- Vary sentence length. A short sentence often lands well after a longer,
  explanatory one.

**Voice**

- Use a formal, neutral, impersonal voice, the voice of a well-edited technical
  book. The reader is not a colleague being spoken to. Avoid "you" and "your",
  and avoid "you can", "you might want to", "simply", "let's", and "we can".
- Prefer impersonal or passive constructions where they read naturally: "The
  metadata is replaced.", "The resulting object is returned.". Do not force the
  passive when it makes a sentence awkward. Clarity outranks any rule here.
- No agentic style, anywhere. No "Let me…", no "we will now…", no
  meta-commentary about the writing, no hedging, none of the tells of
  machine-written text. The prose should read as though a careful engineer
  wrote it by hand.

**References and relationships**

- Make every reference explicit. Do not force the reader to work out what "it",
  "this", "that", "those", "the former", or "the latter" refers to. Repeat a
  noun when the repetition reads more clearly than a pronoun. Repetition is
  preferable to ambiguity.
- Describe a relationship with a verb, not a noun phrase. Write "This function
  reverses the transformation performed by `X`", not "The inverse of `X`".
  Write "This function is the OME-metadata counterpart of `update_attributes`",
  not "The mirror of `update_attributes` for OME metadata".
- Use explanatory verbs — returns, replaces, preserves, combines, removes,
  converts, interprets, validates, delegates — wherever they make the behaviour
  explicit.

**Content**

- Say what the code does, not why or how it is implemented. State the
  behaviour, and lead with the concept and what the caller receives.
  Implementation detail, such as lazy imports, import cycles, caching, or which
  private helper does the work, does not belong in a docstring. Record that
  kind of rationale in a code comment near the implementation.
- Do not narrate history or decisions. Words like "deliberately", "we now", and
  "rather than" describe a changelog. The history belongs in the commit
  message.
- A public docstring is a rendered documentation page. It must not cite an
  issue or PR number or a review finding, name a private symbol the reader
  never touches, or read like a note to a fellow contributor.
- Lead with the simplest spelling. Any runnable example (a `pycon` block) must
  be copy-pasteable, true to what the interpreter prints, and pass on the
  oldest supported Python (3.8).

**Structure of an API docstring**

State, in order: what the function does, then how it behaves in the important
cases, then the essential details such as semantics, edge cases, defaults, and
limitations, and then usage guidance only where it genuinely helps. Do not
begin with implementation unless the implementation is itself the subject of
the documentation.

**Punctuation**

Use punctuation to make prose easier to read, not to compress it. Prefer full
stops. Use em dashes sparingly, and never as a repeated device for structuring
technical information. Do not use semicolons to join independent thoughts.
State information directly rather than in a parenthetical aside. Do not correct
one compression habit by adopting another. Replacing an em dash with a
semicolon, or a long sentence with a list, still forces several ideas into one
place. Write the ideas as separate sentences instead.

**Worked example**

Prefer:

> This function merges the supplied OME metadata with the metadata already
> stored on the node. A value in `ome` replaces the existing value with the
> same top-level key. An existing key that `ome` does not name is preserved.
>
> The merge is shallow. A nested structure, such as a multiscale or plate
> definition, is replaced as a whole rather than merged recursively.

Avoid, as too compressed:

> The mirror of `update_attributes` for OME metadata: the top-level keys of
> `ome` replace those on the node's current OME metadata, and any it already
> has that `ome` does not name are kept.

Avoid, as too conversational:

> This is basically the OME equivalent of `update_attributes`. You can use it to
> merge metadata, while keeping any existing keys that aren't specified.

**Editing existing documentation**

Understand the core of what the text is trying to say, then write the best
version of it. Do not make timid, surface-level edits. If a docstring or
paragraph is muddled, agentic, or over-compressed, rewrite it completely.

**A final read**

Before finishing, read the prose once as a reader would, and confirm that every
sentence is grammatically complete, that every pronoun has one clear referent,
and that no sentence carries too many independent ideas. Confirm that a full
stop has replaced any semicolon or em dash that was there only to compress,
that the reader is not addressed unnecessarily, and that the wording is natural
when read aloud. The result should sound like a person wrote it. Where a
shorter sentence and a clearer sentence disagree, choose the clearer one.

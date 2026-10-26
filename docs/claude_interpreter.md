# Claude reads the sentence (the stronger rule path)

On the Rules page there are now three ways to turn a sentence into a rule:

| Choice | What it does | Cost |
|---|---|---|
| **Offline parser** (default) | a fixed grammar; refuses anything outside it | free, no key |
| **Offline parser + Claude cross-check** | the same grammar, plus Claude can only say no | one paid call per click |
| **Claude reads the sentence** (new) | Claude reads loose wording and fills a strict form; code checks it | one paid call per click |

## What the new path is for

Sentences the offline parser refuses on purpose, for example `alert me when mice attack for more that 2 seconds`
(a typo, and "more than" is not the inclusive minimum a rule uses). Claude reads the typo, maps "more than 2 seconds" to
"at least 2 seconds", and says so in an **assumption** that is shown before you confirm.

## What keeps it safe

Claude only fills the few fields of a rule: behavior, alert mode, minimum duration, and optional thresholds. It never
approves a rule, never touches the monitor or the event log, and never produces scores, counts or events. Code then checks:

1. the reply follows a strict form, or it is thrown away;
2. every quote occurs word for word, once, in your sentence;
3. every number is grounded in the quote: the duration equals the one length of time in its quote ("2 seconds", "two
   seconds", "half a second", "500 ms", "a minute"); a threshold must be written like `alert threshold 0.9`;
4. anything a rule cannot express (a place, a time of day, an "if" condition, exact or maximum durations, negation, other
   behaviors, several rules) blocks the proposal. It is never dropped;
5. an unusual word for a behavior ("fighting") is allowed only with an assumption that says so;
6. words of your sentence that no quote used are listed, so you see what the rule ignored;
7. `rules.validate_rule` has the last word.

You still read the rule and click **Confirm**. The offline parser stays the default.

## Turning it on

Set two variables before starting the dashboard (the key is read from the environment only and is never printed or saved):

```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
$env:LIVE_TRACKER_CLAUDE_MODEL = "<a Claude model id>"
```

The `anthropic` package must be installed in the project's Python. Without a key or a model, the page says "Claude unavailable"
and proposes nothing. One click on Propose rule makes at most one request (plus at most one automatic retry), with a 20 s
time limit and at most 600 output tokens. The sentence is limited to 500 characters.

## What has and has not been tested

- Tested: the checks, the refusals, the page, and the workflow, all with a **fake Claude client** (`tests/test_claude_interpreter.py`,
  `tests/test_interpret_integration.py`). Nothing was sent and nothing was paid.
- **Not tested: how well a real Claude model follows the prompt.** No key was used. The checks make a wrong reading
  fail safely (a refusal or an unavailable message), but they cannot make Claude read well. Try a few real sentences before
  relying on this path, and keep the offline parser as the fallback.
- A guess that passes every check can still be wrong in meaning. That is why every guess is shown as an assumption and
  the researcher confirms.

# How the Rules page reads a sentence

A biologist should not have to write a rule in one fixed form. The Rules page reads the sentence in three ways, from
the simplest to the strongest.

| Choice | What it does | Cost |
|---|---|---|
| **Smart reading** (default) | repairs typos, understands everyday wording, shows what it understood | free, runs on this computer, nothing is sent anywhere |
| **Offline parser + Claude cross-check** | the strict parser plus a second opinion from Claude that can only say no | one paid call per click |
| **Claude reads the sentence** | Claude reads very loose wording and fills a strict form; code checks it (`docs/claude_interpreter.md`) | one paid call per click |

When the smart reading cannot read a sentence, the page says why in plain words, gives an example, and, if an API key is
set, offers a button **Ask Claude to read this sentence**. Claude is never called without a click.

## What the smart reading understands

- **Typos:** "secnds", "attck", "wehn", "more that 2 seconds". A word is repaired only when it is one slip away from exactly
  one word of a small fixed vocabulary; ties and short words are left alone.
- **Polite filler:** "please", "could you", "I would like you to", "thanks".
- **Everyday words for the action:** "let me know", "ping me", "message me", "be notified", "send me an alert" mean *alert me*;
  "write down" means *log*; "confirm with me" means *ask me*.
- **Everyday words for the behavior:** fight, fighting, aggressive, violent, biting mean *attack*; sniffing, exploring mean
  *investigation*; mating means *mount*. Each is shown as an assumption ("Read 'fighting' as attack, the closest behavior the
  classifier scores.").
- **Other ways of saying a minimum:** "more than 2 seconds", "over", "above", "longer than", "2 seconds straight", "3 secs in a row".
  A rule's duration is an inclusive minimum, so "more than 2 seconds" is read as "at least 2 seconds", and the page says so.
- **Word order:** "when the mice fight for more than 5 seconds, let me know".

Every change is listed under **Assumptions to check before confirming**, together with the sentence the rule was really read
from. You still read the rule and click **Confirm**.

## What it refuses on purpose (and says why)

- A missing length ("for a while", "for a few seconds"): it never invents a number of seconds.
- "less than", "under", "at most", "no more than", "up to", "exactly": a rule is a minimum, so these cannot be turned into one.
- A place, a time of day, an "if" condition, a count, negation ("don't alert me", "unless"), another behavior (grooming,
  chasing), or several behaviors in one rule.

## Limits

- It reads English sentences with a closed vocabulary. Very free wording, or a long story, may be refused. Then rephrase, or
  use **Claude reads the sentence**.
- "fighting" to *attack* and "mating" to *mount* are judgments. They are shown as assumptions, and the earlier design
  refused them. The researcher decides when confirming.

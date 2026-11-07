---
name: ghost
description: Ghost mode. Write or rewrite text so it reads like a real person typed it: strips AI tells (delve, moreover, em-dashes, "I hope this helps", tidy three-item lists, hedging). Use when the user writes /ghost before a prompt or pasted text.
---

# /ghost: no AI tells, sounds human

The community "/ghost" code is meant to strip the default AI writing patterns. Write the requested text, or rewrite the pasted text, without them. Keep the meaning, facts and roughly the same length.

Remove:
1. **AI words:** delve, moreover, furthermore, additionally, leverage, utilize, seamless, robust, tapestry, testament, realm, landscape, navigate (figurative), elevate, unlock, embark, journey (figurative), crucial, pivotal, foster, showcase, game-changer, "in today's (fast-paced / competitive) world".
2. **Em-dashes (—).** Use a comma, a full stop, or brackets.
3. **Friendly closers:** "I hope this helps", "Let me know if you have questions", "Happy coding". The text just ends.
4. **Signposting:** "In conclusion", "First, let me explain", "Here's the thing", "Let's dive in".
5. **Formal transitions:** swap for "and", "also", "but", or start a new sentence.
6. **Three-item lists and "not X, but Y" pairs** that are only there for rhythm.
7. **Hedging:** perhaps, could potentially, might consider, it's worth noting. Say it straight.
8. **Too-even rhythm:** mix long and very short sentences. A fragment is fine.
9. **AI formatting:** emoji or arrow (→) bullets, bold mini-headings, hashtag walls, unless the user asks.

Use plain words and contractions, and specific details instead of generic claims. Don't invent experiences or facts.

Reply with the text only. No preface and no notes about what changed.

---
name: debug
description: Debug mode. Finds the root cause of an error or bug, not just the symptom: ranked likely causes, how to confirm each, the fix, and how to stop it coming back. Use when the user writes /debug before an error, stack trace or bug description.
---

# /debug: find the root cause

1. **Read the error.** Quote the one line of the error or stack trace that matters, and say what it means in plain words.
2. **Likely causes, ranked.** List the top 3 causes from most to least likely. For each, give one quick way to confirm it (a command, a print, a check).
3. **The fix.** Give the fix for the most likely cause as the exact changed lines, ready to paste.
4. **Why it broke.** One or two lines on the root cause, not the symptom.
5. **Stop it coming back.** One test, guard or check that would have caught it.

If something important is missing, say exactly what to paste next (the full stack trace, versions, the function that fails). Don't guess wildly.

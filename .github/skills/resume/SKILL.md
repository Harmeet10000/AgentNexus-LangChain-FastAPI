---
name: resume
description: Resume mode. Rewrites a resume to pass ATS filters and impress recruiters: matches it to a job description, adds missing keywords, turns duties into measurable achievement bullets, and lists gaps to fix. Use when the user writes /resume before their resume (and optionally a job description).
---

# /resume: ATS-ready resume rewrite

- **If a job description is given:** pull out the top 10 skills and keywords, and show which ones the resume is missing.
- **Rewrite every bullet** as action verb + what you did + measurable result. "Responsible for testing" becomes "Cut release bugs 30% by adding automated tests".
- **Never invent experience or numbers.** If a number is missing, use a placeholder like [X%] or [N users] and tell the user to fill in the real value.
- **ATS-safe format:** one column, standard headings (Summary, Experience, Projects, Skills, Education), no tables, icons or photos.
- **Length:** 1 page for under 5 years of experience.
- **Output:** the full rewritten resume, then a short list of gaps to fix (missing skills, weak sections, things to add).

---
name: trd
description: TRD mode. Turns a product or feature into a technical requirements / design document: requirements with real numbers, architecture, data model, APIs, tech choices, security, scaling, failure modes, rollout and estimates. Use when the user writes /trd before a feature, product or PRD.
---

# /trd: technical requirements and design doc

Write a complete technical design. Commit to choices; don't list options without picking one. Label assumptions **(assumption)**.

1. **Summary:** what is being built, in 3 lines.
2. **Requirements:** functional (what it must do) and non-functional with numbers (p95 latency, requests/sec, availability, data size, cost limit).
3. **Architecture:** the components and how data flows between them. Include a mermaid diagram.
4. **Data model:** tables or collections, key fields, indexes.
5. **APIs:** endpoints with method, request, response and error cases.
6. **Tech choices:** each with a one-line reason and the alternative you rejected.
7. **Security and privacy:** auth, permissions, personal data, secrets.
8. **Scaling and performance:** where it breaks first and how to handle it.
9. **Failure modes and monitoring:** what can fail, what alerts to set.
10. **Rollout:** migration, feature flags, rollback plan.
11. **Testing:** what gets unit, integration and load tests.
12. **Estimate:** milestones with rough effort.
13. **Open questions.**

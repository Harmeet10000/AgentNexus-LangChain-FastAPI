# The Zero Trust Gap in LLMs

The speaker emphasizes that LLM attacks have moved from being anomalies to the baseline. These systems suffer from a lack of native separation between system controls and data, leading to a "zero trust gap." The six primary attack vectors identified are:

**Prompt Injection** (1:15): Direct user inputs designed to override system instructions and exfiltrate confidential data (e.g., the Sydney/Bing case).
**Context/Indirect Injection** (3:26): Adversarial instructions hidden in external content (websites, emails) that the LLM processes, bypassing the user interface.
**Model Internals**(6:03): Exploiting the model’s mathematical nature by finding "gibberish" suffix tokens that break alignment and force the model to ignore safety refusals.
**RAG Poisoning**(9:28): Inserting malicious chunks into knowledge databases that retrieval-augmented generation systems then fetch and prioritize.
**Model Context Protocol (MCP) Exploits** (10:54): Leveraging asymmetry between what a user sees and what an LLM reads, often using hidden parameters to steal credentials.
**Agentic Escalation** (12:04): Autonomous agents tricked into performing harmful actions, such as downloading and executing malicious files or installing compromised packages.
**ModernBERT**: The Defensive Secret Sauce
The speaker advocates for using encoder-only models like ModernBERT instead of "LLM-as-a-Judge" for safety checks, citing superior latency (approx. 35ms) and the ability to process the full context in one pass (18:13).

## Why Encoders Models

**Bidirectional Attention**: Full Context understanding
**Speed**: <35ms baseline
**Efficiency**: Fine Tune in a Singe GPU(in hours)
**Privacy**: Self Hosted
**Scale vs LLM-as-a-Judge**: 1req/sec @ token cost

## Key architectural features that make ModernBERT efficient include

**Alternating Attention** (22:04): Switches between local (sliding window) and global attention, mimicking human reading patterns to handle long contexts (up to 8,192 tokens) without quadratic scaling issues.
**Unpadding & Sequence Packing** (25:17): Eliminates wasted computation on "padding" tokens, allowing for efficient processing of variable-length inputs in a single forward pass.
**Rotary Positional Encoding (RoPE)** (30:48): Rotates query/key projections based on position rather than adding fixed vectors, maintaining semantic integrity and supporting continuous context windows.
**FlashAttention** (33:28): A hardware-level optimization that keeps computation in ultra-fast on-chip GPU memory, significantly reducing the bottleneck caused by memory transfers.
**reduces the TC to O(n*window_size) as compared to O(n2)**

## Practical Implementation

The talk provides a walkthrough for building a safety discriminator:

**Fine-tuning**: Uses a dataset (e.g., Inject) to train a binary classification head on top of the encoder output (the CLS token) InjectGuard Dataset 75K samples from 20 open sourced. Install FlashAttention
**Efficiency**: By using optimizations like bfloat16 and Adam optimizers(adamw_torch_optimizer), the speaker achieved ~85% (with BIPIA, NotInjet, Wildguard-Benig and PINT)accuracy with sub-50ms latency.
**Conclusion**: The speaker encourages developers to build their own defensive layers on commodity hardware rather than relying solely on external model alignment, emphasizing that safety is a shared, ongoing responsibility (43:24).


## The 4-Step Maturity Model
Step 1: Ad Hoc (4:31 - 5:10): The initial stage where systems are built without formal risk management or security considerations.
Step 2: Foundation (5:12 - 7:22): Establishing the basics, including assigning non-human identities to agents, enabling delegation/on-behalf-of flows, and using a SIEM (Security Information and Event Management) for auditability.
Step 3: Enhanced (7:29 - 10:46): Treating agents as first-class citizens with ephemeral credentials, applying fine-grained, context-based access, and implementing real-time detection for anomalies.
Step 4: Adaptive (10:47 - 14:14): The most advanced stage involving continuous authentication, risk-based re-authentication, and real-time revocation to dynamically secure non-deterministic workflows.

1. JIT Permissions - Over-Permissioning - Force point-of-use enforcement: Verify policy compliance at the exact moment of connection to sensitive data.
2. IAM
3. hidden prompt & prompt injection proof
4. tool security - tool injection protection
5. sandbox environment
6. MCP security - Use a secure vault to manage tool credentials, providing only temporary access rather than storing long-term secrets within the MCP server
7. TTL based tokens - Occurs when tokens are intercepted or leaked (e.g., through LLM prompts). Use tokens that represent both the user (subject) and the agent (actor) to validate that an agent is authorized to work on the user's behalf. Use token exchange at each hop of a flow to ensure security propagates through the system, and restrict tokens to specific scopes to enforce the principle of least privilege
8. <https://blog.langchain.com/agent-authorization-explainer/>
9. guardrails
10. TLS - Utilize TLS/MTLS to prevent man-in-the-middle attacks and ensure all stored credentials are encrypted
11. Ensure audit logs record when a human specifically tasks an agent with an action.

## Agent Sandbox

the necessity of dedicated, secure infrastructure—or an "Agent Computer"—for running autonomous AI agents. He argues that traditional containerization is insufficient for untrusted agent-generated code and outlines the technical requirements for robust sandboxing.

1. Beyond Localhost (1:25 - 2:49): Naresh establishes that long-running agents require a purpose-built environment that maintains state, networking, and connectivity between the internet and the agent's workspace.
2. The 5 Infrastructure Layers (2:49 - 4:30): He breaks down the sandbox architecture into five critical components: isolation boundary (MicroVM), containers (e.g., Ubuntu), bash sessions/file system, networking (port routing), and persistence.
3. Runtime Decisions (4:30 - 7:10): A discussion on providing agents with full Linux instances rather than restricted shells. Key capabilities include managing background processes (like dev servers), streaming output, and supporting multiple concurrent sessions.
4. The Security Paradox (7:10 - 9:57): Naresh addresses the danger of executing untrusted code. He explains that Docker (namespaces) is a security mismatch because it shares the host kernel. He advocates for MicroVMs (Firecracker), which provide hardware-level isolation with minimal overhead.
5. Networking Architecture (9:57 - 11:37): Explanation of how to dynamically route internet traffic to specific, ephemeral sandboxes using unique session tokens and internal mapping, ensuring browser-to-agent connectivity.
6. Persistence & Cold Starts (11:37 - 15:53): A deep dive into strategies to eliminate cold starts, including pre-built images, persistent volumes, memory snapshots, and maintaining "warm pools" of ready-to-use sandboxes.
7. The 8-Line Production Sandbox (15:53 - 18:08): A conceptual breakdown of the minimal code required to orchestrate a secure, persistent, and network-accessible sandbox.
8. Scale of AI Code (18:08 - 18:48): The speaker notes the massive scale of AI-generated code (e.g., billions of lines daily on Cursor) and why infrastructure for these agents is becoming the new default.
9. Future of Runtimes (18:48 - 20:44): Discussion on moving toward even lighter solutions like V8 isolates for specific workloads where a full MicroVM may be overkill.
10. Q&A and Programmatic Gates (20:44 - 29:57): The speaker covers standardizing agent harnesses and the use of programmatic gates—a strategy where agents are given limited, specific pathways for interaction to control the "blast radius" of their actions.

### Key Takeaways for Secure Sandboxing

1. Isolation: Use MicroVMs (Firecracker) rather than standard Docker containers. MicroVMs provide a dedicated kernel per sandbox, preventing an agent from impacting the host machine (8:33).
2. Blast Radius Control: Rather than giving agents broad permissions, implement programmatic gates. These act as strictly defined interfaces that only allow the agent to perform specific, approved actions (25:48).
3. Persistence Strategy: To ensure a seamless user experience, leverage memory snapshots and warm pools. This allows the agent to resume work instantly without waiting for the environment to boot (12:51).
4. Environment over Behavior: Naresh emphasizes that the most effective security measure is to control the environment itself. By ensuring the agent operates within an isolated sandbox with no sensitive network or file system access, you mitigate the risks associated with autonomous code execution (28:35).

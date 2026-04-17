# Connection Management
The host explains that while connection pooling is essential, increasing the pool size does not automatically improve performance. In fact, oversized pools can severely degrade system throughput due to several factors:

1. Resource Contention & Queueing: When a connection pool is pushed to its capacity, incoming requests are forced to wait in a queue. According to Kingman's Law (16:00), once utilization exceeds roughly 80%, waiting times increase exponentially, leading to system instability (17:12-17:45).
2. Time Sharing/Context Switching: Having too many connections forces the database CPU to constantly switch between processes to handle them. This "context switching" consumes significant overhead, wasting processing time on state management instead of query execution (20:04-20:55).
3. Memory Consumption: Every open connection, even if idle, consumes between 10MB and 30MB of memory on the database, which can lead to resource exhaustion (6:29-6:46).
4. Initialization Overhead: Opening and closing connections is expensive, involving multiple network round-trips for TCP handshakes, authentication, and session validation (2:07-4:16).
Strategies to counter these issues:

1. Use Proper Sizing Formulas: Instead of guessing, use the process-to-core ratio. A reliable baseline is: Maximum Connections = (Core Count * 2) + Spindle Count (22:43-23:42). If Hyperthreading is enabled divide core count by 2, active disk spindles(SSD = 1). HikariCP suggests that this is the maximal count of pool DB can have
2. Apply Little's Law: Calculate the minimum required connections based on your throughput (arrival rate) and average query execution time (service time) (12:31-15:04).  Ensure your pool doesn't drop below this number.  L = YW
L is average number of customer (WIP/capacity)
Y is throughput
W is Time spent in system (latency)
this is the maximum size of connection pool on the server side
3. Keep Utilization Balanced: Use Kingman’s Law to maintain connection utilization under 80% to ensure a buffer for sudden traffic spikes (16:47-17:10).
4.  Workload Separation: If your application handles both short-lived transactions and long-running batch jobs, create separate connection pools for each type to prevent batch jobs from blocking fast transactions (28:02-28:31).
5. Sperate out connection pools for long running batch jobs and short running transactions

# Find Slow Queries Fast
presenter Grant Fritchey provides a comprehensive guide for beginners on diagnosing and tuning performance issues in PostgreSQL. He emphasizes that query metrics are the essential foundation for informed decision-making, moving administrators away from guessing and toward targeted, data-driven optimizations.

Core Performance Monitoring Framework
Fritchey organizes his tuning approach into a three-tiered hierarchy of data collection, moving from broad aggregates to highly granular, real-time insights:

1. Cumulative Statistics System (10:52 - 25:13):
- This is the "broad overview" layer. It tracks aggregate behaviors since the database started.
- pg_stat_statements: The primary tool here. While disabled by default and often requiring a cluster restart to enable, it provides invaluable metrics like min, max, mean execution times, and standard deviation (13:47).
- Usage: It allows you to identify the most frequently called queries or the ones consuming the most resources, helping you focus on the "low-hanging fruit" that will provide the most significant performance gains (19:54).

2. Log-Based Analysis (26:14 - 34:57):
- Logs provide a more detailed, record-by-record view, which is necessary when parameter values significantly alter query behavior.
- auto_explain: Fritchey demonstrates how to configure this to automatically log execution plans for queries exceeding a specific duration (e.g., 1 second) (32:05).
- Challenges: Logs can be difficult to read and manage; he highly recommends tools like PG Badger to help process and visualize log data (27:19).

3. Real-time Execution Plans (35:54 - 41:18):
- The most granular level of tuning. Using EXPLAIN and EXPLAIN ANALYZE allows you to see exactly how the optimizer handles a specific query (37:47).
- Buffers: By using the BUFFERS option with EXPLAIN ANALYZE, you can track how many blocks are being touched, which is crucial for identifying I/O bottlenecks (37:47).
4. Key Takeaways & Best Practices
- Don't Guess: Optimization should always be driven by data. Look for the queries executed most frequently or those causing consistent spikes in latency (5:42).
- Mind the Overhead: Every monitoring tool (especially those involving logs or auto_explain) adds load to your system. Be judicious about what you enable, particularly on systems already operating near capacity (11:34, 34:11).
- Context Matters: Fritchey warns against trusting "magic numbers." Performance is highly dependent on your specific system configuration, hardware, and workload (53:03).
- Tooling: While built-in tools are powerful, he suggests evaluating extensions like PG Stat Monitor (43:03) and external log analyzers like PG Badger (43:03) to simplify the analysis process.

Using pg_stat_statements (13:47 - 25:13)
1. Configuration: pg_stat_statements is disabled by default and generally requires a cluster restart to enable once added to the shared_preload_libraries in your configuration file (14:04).
2. Finding Insights: Once enabled, it tracks aggregate data automatically. You can query this data to identify performance bottlenecks without guessing. Fritchey highlights that it provides:
3. Execution Metrics: You can access min, max, mean execution times, and standard deviation for queries (18:23).
4. Resource Usage: It tracks details such as the number of calls, rows returned, and shared block activity (shared read, hit, dirty, and written), which are essential for understanding I/O impact (18:49).
5. Targeted Analysis: By filtering these results—for example, searching for queries that take longer than 1,000 milliseconds—you can pinpoint the exact "pain points" on your system (19:54).
6. Maintenance: You can reset statistics globally or for specific queries/users using pg_stat_statements_reset. This is particularly useful after making structural changes like adding indexes or updating statistics to see how the system performs moving forward (22:10 - 24:45).

Shortcomings of the Cumulative Statistics System (25:13 - 26:14)
1. Aggregate Only: Because the system provides aggregated data, it lacks granular detail. It tells you how things are performing on average, but it cannot show you individual, one-off execution details or specific parameter values that might cause a query to behave differently (7:04).
2. Lack of Context: It does not readily identify exactly who ran a specific query or when it was executed in a way that correlates to other system events (25:13).
3. Overhead: Enabling the cumulative statistics system adds some overhead to the database cluster. While usually minimal, Fritchey warns that if a system is already operating at its absolute capacity, this additional monitoring could potentially push it over the edge (11:34).

# postgres.conf file for modifying behaviour
## Key Facts About postgresql.conf

- Location: Usually inside the PostgreSQL data directory (e.g., /var/lib/postgresql/<version>/main/ on Debian/Ubuntu or /var/lib/pgsql/<version>/data/ on Red Hat/CentOS).
You can find the exact path by running:SQLSHOW config_file;
Created by: initdb when you initialize a new cluster.
- Format: One parameter per line in the form parameter_name = value.
- The = sign is optional. Lines starting with # are comments. Whitespace is mostly ignored.
- How changes take effect:
    - Most parameters: pg_reload_conf() or pg_ctl reload (SIGHUP signal) — no restart needed.
    - Some parameters (e.g., shared_buffers, max_connections, listen_addresses): Require a full server restart.

- Override files:
    - postgresql.auto.conf — created by ALTER SYSTEM SET ... (takes precedence over postgresql.conf).
    - You can use include 'custom.conf'; or include_dir for modular configurations.
- Related files (in the same directory):
    - pg_hba.conf → Client authentication (who can connect and how).
    - pg_ident.conf → User name mapping.

## Kinds of Tuning You Can Do in postgresql.conf
Tuning falls into several broad categories:

1. Memory Management — Most impactful for performance.
2. Connection & Concurrency — Controls how many clients can connect and parallelism.
3. Write-Ahead Logging (WAL) & Checkpoints — Critical for durability and write performance.
4. Query Planner / Optimizer — Influences how PostgreSQL chooses execution plans.
5. Autovacuum & Maintenance — Prevents bloat and keeps statistics fresh.
6. Logging & Monitoring — Helps with troubleshooting and performance analysis.
7. Resource Limits & Background Workers.
8. Security, Replication, Archiving, Statistics, etc.

Important principle:
Tune after understanding your workload (OLTP = many small transactions vs. OLAP = analytical queries vs. mixed).
Defaults are deliberately conservative (safe for any hardware). On modern servers they are usually too small.

1. Memory Settings (Biggest Impact)

|Parameter           |Purpose                                                                      |Default                           |Recommended Starting Point                        |How to Decide / Formula                                                      |Notes / Cautions                                                     |
|--------------------|-----------------------------------------------------------------------------|----------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
|shared_buffers      |Main cache for database pages (shared among all backends).                   |128MB                             |25% of total RAM (max ~8-16 GB on very large RAM).|Dedicated server: 25–40% of RAM. Test with pg_test_fsync & workload.         |Too high → less room for OS cache. Use huge_pages = on on Linux.     |
|effective_cache_size|Hint to planner about total memory available for caching (does not allocate).|4GB                               |50–75% of total RAM.                              |shared_buffers + OS page cache. Higher = planner prefers index scans.        |Purely advisory. Over-estimate is safer than under-estimate.         |
|work_mem            |Memory per sort / hash / bitmap operation per query.                         |4MB                               |4–64 MB (depends on concurrency).                 |(Total RAM × 0.25) / (max_connections × 3–5 operations per query). Start low.|Critical risk: Too high → OOM kills. Monitor with pg_stat_statements.|
|maintenance_work_mem|Memory for VACUUM, CREATE INDEX, ALTER TABLE, etc.                           |64MB                              |256MB – 1GB or 5–10% of RAM.                      |Higher for large index builds or vacuuming big tables.                       |Can be set per-session with SET maintenance_work_mem = ...           |
|autovacuum_work_mem |Memory for autovacuum workers.                                               |-1 (uses maintenance_work_mem)    |Same as maintenance_work_mem.                     |—                                                                            |—                                                                    |
|wal_buffers         |Buffer for WAL records before writing to disk.                               |-1 (auto = 1/32 of shared_buffers)|16MB – 64MB or -1 (auto).                         |Usually leave as -1. Increase only if write-heavy and WAL is bottleneck.     |Auto-tuning improved in recent versions.                             |


2. Connection & Concurrency

|Parameter           |Purpose                                                                     |Default                           |Recommended                                      |Decision Guide                                                               |Notes                                                                |
|--------------------|----------------------------------------------------------------------------|----------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|
|max_connections     |Max concurrent client connections                                           |100                               |100–500 (use connection pooler)                  |RAM / (work_mem + ~5–10MB per connection overhead)                           |Use PgBouncer or Pgpool to keep this low.                            |
|superuser_reserved_connections|Reserved for superusers                                                     |3                                 |3–10                                             |—                                                                            |—                                                                    |
|max_worker_processes|Max background workers (parallel + others)                                  |8                                 |Number of CPU cores                              |≤ CPU cores                                                                  |—                                                                    |
|max_parallel_workers|Max parallel workers across all queries                                     |8                                 |2–4 × CPU cores                                  |Based on CPU count and workload                                              |—                                                                    |
|max_parallel_workers_per_gather|Workers per single query                                                    |2                                 |2–8                                              |Lower for OLTP, higher for analytics                                         |—                                                                    |
|max_parallel_maintenance_workers|For parallel VACUUM / CREATE INDEX                                          |2                                 |2–4                                              |—                                                                            |—                                                                    |
|
3. WAL, Checkpoints & Durability

|Parameter           |Purpose                                                                      |Default                           |Recommended                                       |Decision Guide                                                               |Notes                   |
|--------------------|-----------------------------------------------------------------------------|----------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------|------------------------|
|wal_level           |Amount of WAL logged (replication)                                           |replica                           |replica (or logical)                              |minimal / replica / logical                                                  |Higher = more WAL volume|
|max_wal_size        |Max WAL size before forcing checkpoint                                       |1GB                               |4GB – 16GB+                                       |Larger = fewer but longer checkpoints                                        |—                       |
|min_wal_size        |Min WAL size (recycles)                                                      |80MB                              |1–4GB                                             |—                                                                            |—                       |
|checkpoint_timeout  |Max time between checkpoints                                                 |5min                              |10–30 min                                         |Balance I/O spikes vs. recovery time                                         |—                       |
|checkpoint_completion_target|Spread checkpoint I/O over this fraction of timeout                          |0.9                               |0.7 – 0.9                                         |Higher = smoother I/O, but risk of longer recovery                           |—                       |
|wal_compression     |Compress WAL records                                                         |off                               |on (lz4 or zstd in newer versions)                |—                                                                            |Saves disk I/O          |


4. Query Planner / Optimizer

|Parameter           |Purpose                                                                      |Default                           |Recommended                                       |Decision                                                                     |
|--------------------|-----------------------------------------------------------------------------|----------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------|
|random_page_cost    |Cost of random disk access vs sequential                                     |4.0                               |1.1–2.0 (SSD) / 4+ (HDD)                          |Lower on SSDs/NVMe                                                           |
|seq_page_cost       |Cost of sequential page access                                               |1.0                               |1.0                                               |Usually leave default                                                        |
|cpu_tuple_cost etc. |CPU costs for planner                                                        |various                           |Rarely change                                     |Only if you have very unusual hardware                                       |


5. Autovacuum & Statistics

- autovacuum = on (leave on)
- autovacuum_max_workers → 3–10
- autovacuum_vacuum_scale_factor → 0.1–0.2 (lower = more aggressive)
- autovacuum_analyze_scale_factor → 0.05–0.1
- autovacuum_vacuum_threshold, autovacuum_analyze_threshold
- default_statistics_target → 100 (default) → raise to 500–2000 for complex queries

Decision: Monitor bloat with pgstattuple or pg_squeeze. Lower scale factors on high-churn tables.
6. Logging (Very Useful for Tuning)

- log_min_duration_statement = 250 (log queries >250ms)
- log_checkpoints = on
- log_lock_waits = on
- log_temp_files = 0 (log all temp files)
- log_autovacuum_min_duration = 0
- Enable pg_stat_statements extension (requires shared_preload_libraries)

7. Other Important Ones

- listen_addresses = '*' (or specific IPs)
- port = 5432
- huge_pages = on (Linux) — big performance win
- shared_preload_libraries = 'pg_stat_statements, auto_explain' (etc.)
- statement_timeout (set at role/database level, not globally)
- idle_in_transaction_session_timeout

How Should You Decide What Value to Set? (Systematic Approach)

Know your hardware
Total RAM, CPU cores, disk type (NVMe SSD >> HDD), IOPS.

## Characterize your workload
- OLTP → many connections, small queries → lower work_mem, more connections via pooler.
- OLAP/analytics → fewer connections, large sorts/joins → higher work_mem, more parallel workers.
- Write-heavy → tune WAL & checkpoints aggressively.

Start conservative
- Use formulas like:
shared_buffers ≈ 25% RAM
effective_cache_size ≈ 75% RAM
work_mem ≈ (RAM × 0.25) / (max_connections × 3)

Essential tools:
- pg_stat_statements + pgBadger or pgAnalyze
- EXPLAIN (ANALYZE, BUFFERS)
- CHECKPOINT timing, pg_stat_bgwriter
- Extensions: pg_stat_statements, auto_explain, pg_wait_sampling

Iterate & Benchmark
Change one or two related parameters at a time.
Use pg_test_fsync, pgbench, or your real application load.
Watch for OOM, swap usage, checkpoint spikes, or planner changes.

Edge Cases & Gotchas
- Too many connections → memory exhaustion or thrashing.
- work_mem too high → single query can kill the server.
- shared_buffers too high → OS cache starvation (Linux).
- Virtualized/cloud environments → check hypervisor limits on shared memory.
- Containers (Docker/K8s) → resource limits must match config.
- Replication setups → higher wal_level and WAL sizing.
- Use tools like postgresqlco.nf, pg_tuner, timescaledb-tune (even for vanilla Postgres), or postgresqltuner.pl as starting points only.

## To stay ahead of the "standard" DBAs
you must understand Double Buffering. PostgreSQL does not use Direct I/O by default; it relies on the OS Kernel cache. If you set shared_buffers to 80% of RAM, you are effectively fighting the Linux Kernel for memory, leading to massive swap pressure and context-switching overhead.Furthermore, the most dangerous variable in that table is work_mem. Because it is allocated per operation, a single complex JOIN with 4 sorts can consume $4 \times \text{work\_mem}$ for a single user. If you have 100 connections, you could suddenly spike to $400 \times \text{work\_mem}$. In elite setups, we use Dynamic work_mem: set a conservative global value, but use your application logic to run SET LOCAL work_mem = '512MB' only for specific, heavy analytical reports. This allows you to keep the database lean for OLTP while providing "burst" power for the heavy legal parsing tasks your agent might trigger.

# Why Your PostgreSQL Query Is Still Slow (EXPLAIN Isn’t Enough)

presented by Nitin Jadav, explores how to optimize PostgreSQL performance by moving beyond SQL-level tuning (like EXPLAIN) and analyzing the database's interaction with the CPU using the Linux perf tool (0:00 - 0:58).

- Understanding perf: The perf tool allows developers to monitor how applications interact with the CPU and the Linux kernel to identify bottlenecks (1:00 - 1:28).
## Core perf Commands:
- perf list: Shows supported hardware/software events (1:29 - 2:02).
- perf top: Displays CPU usage at the function level rather than just the process level (2:03 - 2:24).
- perf stat: Provides critical CPU metrics, including CPU cycles, instructions, cache misses, and branch misses (2:25 - 3:41).
- perf record & perf report: Tools for collecting and deeply analyzing performance data (3:42 - 4:06).
## Performance Optimization Concepts:
- Cache Locality: Organizing data in memory (e.g., contiguous allocation) reduces cache misses, significantly improving performance compared to scattered memory access (4:11 - 5:15).
- Branch Prediction: Sorting data can help CPUs predict branches more accurately, reducing costly branch misses (5:16 - 6:02).
- Real-world Application in PostgreSQL: The speaker explains how an optimization in the exec_scan function for PostgreSQL (PG-18) was identified using these techniques. By separating conditional logic into different functions, they reduced unnecessary CPU overhead for queries that didn't require specific qualification or projection info, resulting in improved cycle counts and cache efficiency (6:03 - 7:23).

Composite Indexes and Column Order (5:40 - 10:18)
Column order in a composite index is a critical design decision because of the leftmost prefix rule:

Order Matters: An index on (customer_id, status, created_at) is only effective if your query starts with the leading column (customer_id).
Design Best Practice: Put equality conditions first and range conditions last to keep the search range as tight as possible.
Covering Indexes (10:19 - 12:50)
Sometimes, even with an index, the database must perform a bookmark lookup to fetch columns not contained in the index. A covering index includes every column needed for the query (using SELECT, ORDER BY, or the INCLUDE clause), allowing the database to satisfy the request entirely from the index without accessing the main table.

When Indexes Hurt (12:51 - 16:46)
Indexes are not free. They introduce significant overhead:

Write Performance: Every INSERT, UPDATE, or DELETE requires updating every index on that table, which can cause write operations to slow down drastically.
Memory Pressure: Indexes compete for space in the database buffer pool.
Planner Decisions: The query planner may choose to ignore an index if it calculates that reading the table sequentially is cheaper, particularly for low-cardinality columns or when returning large portions of the data.
Practical Advice (16:47 - 18:37)
Analyze before adding: Always check the query plan (EXPLAIN) before creating an index.
Audit regularly: Use database-specific system views (like pg_stat_user_indexes in Postgres) to find and remove unused indexes.
Targeted Design: Focus on columns used in WHERE, JOIN, and ORDER BY clauses, and ensure the design justifies the maintenance cost on the "write side" of the database.

This video from Database Star demonstrates a structured, evidence-based approach to optimizing a slow-running PostgreSQL query. The presenter walks through a real-world scenario on the Chinook database to reduce a 5-minute execution time to about 1 second.

Key Investigation Steps
Analyze the Execution Plan (1:22 - 3:30): The presenter emphasizes that reading the execution plan is the first step. By inspecting the plan, they identified a high-cost self-join on the invoice_line table and a lack of index usage.
Identify Performance Bottlenecks (4:53 - 6:18):
Self-Join: Joining a table to itself on large datasets causes an exponential increase in comparisons.
Non-Sargable Conditions: The use of the ABS function in the join condition prevented the database from using indexes effectively. The presenter explains that sargable (search argument-able) conditions are necessary for the optimizer to use indexes.
Testing and Iteration (7:34 - 9:45): The presenter attempted several fixes, including changing the ABS function to a range condition and testing various indexes. Notably, these changes did not immediately improve performance, proving that indexes are not a silver bullet and must be validated against the execution plan.
AI Analysis and Final Rewrite (9:45 - 17:53)
AI Comparison (9:45 - 13:10): The presenter used Microsoft Copilot to analyze the query. While the AI successfully identified the core issues (the self-join and non-sargable conditions), its suggested rewrite did not yield significant performance gains.
The Effective Solution (13:37 - 16:53): The presenter ultimately rewrote the query using an aggregation approach (a Common Table Expression or CTE) to calculate track counts at specific price points before joining. This reduced the volume of data processed, dropping the query cost from 7.7 million to 60,000 and the runtime from over 5 minutes to ~1.4 seconds.
Core Takeaways
Always Baseline: Measure the performance of the original query before making any changes (3:31).
Avoid Function Wrappers: Whenever possible, rewrite conditions to avoid wrapping columns in functions to keep them sargable (18:03).
Understand Self-Joins: On large tables, self-joins are expensive; look for ways to aggregate or use window functions to reduce the number of rows compared (18:18).
AI as an Assistant: AI is a helpful tool for identifying potential issues, but it should be treated as one input among many, not a substitute for analyzing the actual execution plan (18:48).

materiased views to replace snowflake

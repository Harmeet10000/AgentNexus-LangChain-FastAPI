# EXPLAIN (ANALYZE, BUFFERS) — bm25_search__sort_off

```
Limit  (cost=30.65..1811.80 rows=50 width=64) (actual time=21.495..21.508 rows=50.00 loops=1)
  Buffers: shared hit=25979
  ->  Result  (cost=30.65..29419.52 rows=825 width=64) (actual time=21.494..21.503 rows=50.00 loops=1)
        Buffers: shared hit=25979
        ->  Incremental Sort  (cost=30.65..25280.09 rows=825 width=349) (actual time=21.492..21.495 rows=50.00 loops=1)
              Sort Key: ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query)), id
              Presorted Key: ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query))
              Full-sort Groups: 1  Sort Method: quicksort  Average Memory: 57kB  Peak Memory: 57kB
              Buffers: shared hit=25979
              ->  Index Scan using chunks_bm25_idx on chunks c  (cost=0.01..25243.00 rows=825 width=349) (actual time=10.571..21.426 rows=65.00 loops=1)
                    Order By: (search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query)
                    Filter: (((user_id)::text = 'scratch-user-00'::text) AND ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query) < '0'::double precision))
                    Rows Removed by Filter: 1332
                    Index Searches: 1
                    Buffers: shared hit=25973
Planning:
  Buffers: shared hit=270
Planning Time: 1.136 ms
Execution Time: 21.566 ms
Allocated Memory: allocated_by_plan=2708kB allocated_by_exec=1787kB base_allocation=4092kB
```

## Session settings for this capture

`SET LOCAL enable_sort = off` was issued on the same connection immediately before
`EXPLAIN (ANALYZE, BUFFERS)` (see the capture script; `EXPLAIN` output does not echo
session GUCs). The companion `bm25_search__default_cost.md` ran the identical statement
without that setting. Compare the two for the forced-vs-default plan choice.

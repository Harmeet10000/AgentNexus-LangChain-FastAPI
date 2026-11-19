# EXPLAIN (ANALYZE, BUFFERS) — bm25_search__default_cost

```
Limit  (cost=30.65..1811.80 rows=50 width=64) (actual time=21.981..21.994 rows=50.00 loops=1)
  Buffers: shared hit=25979
  ->  Result  (cost=30.65..29419.52 rows=825 width=64) (actual time=21.980..21.989 rows=50.00 loops=1)
        Buffers: shared hit=25979
        ->  Incremental Sort  (cost=30.65..25280.09 rows=825 width=349) (actual time=21.977..21.980 rows=50.00 loops=1)
              Sort Key: ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query)), id
              Presorted Key: ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query))
              Full-sort Groups: 1  Sort Method: quicksort  Average Memory: 57kB  Peak Memory: 57kB
              Buffers: shared hit=25979
              ->  Index Scan using chunks_bm25_idx on chunks c  (cost=0.01..25243.00 rows=825 width=349) (actual time=10.822..21.904 rows=65.00 loops=1)
                    Order By: (search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query)
                    Filter: (((user_id)::text = 'scratch-user-00'::text) AND ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query) < '0'::double precision))
                    Rows Removed by Filter: 1332
                    Index Searches: 1
                    Buffers: shared hit=25973
Planning:
  Buffers: shared hit=270
Planning Time: 1.171 ms
Execution Time: 22.057 ms
Allocated Memory: allocated_by_plan=2708kB allocated_by_exec=1787kB base_allocation=4089kB
```

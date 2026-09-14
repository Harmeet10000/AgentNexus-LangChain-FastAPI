# EXPLAIN (ANALYZE, BUFFERS) — vector_search

```
Limit  (cost=109.90..385.10 rows=50 width=64) (actual time=38.086..120.533 rows=50.00 loops=1)
  Buffers: shared hit=27673
  ->  Incremental Sort  (cost=109.90..13737.50 rows=2476 width=64) (actual time=38.085..120.527 rows=50.00 loops=1)
        Sort Key: ((embedding <=> '[<768-float embedding literal>]'::vector)), id
        Presorted Key: ((embedding <=> '[<768-float embedding literal>]'::vector))
        Full-sort Groups: 2  Sort Method: quicksort  Average Memory: 29kB  Peak Memory: 29kB
        Buffers: shared hit=27673
        ->  Index Scan using chunks_embedding_idx on chunks c  (cost=104.43..13626.08 rows=2476 width=64) (actual time=18.227..120.451 rows=51.00 loops=1)
              Order By: (embedding <=> '[<768-float embedding literal>]'::vector)
              Filter: ((embedding IS NOT NULL) AND ((user_id)::text = 'scratch-user-00'::text))
              Rows Removed by Filter: 596
              Index Searches: 0
              Buffers: shared hit=27667
Planning:
  Buffers: shared hit=258
Planning Time: 1.025 ms
Execution Time: 120.987 ms
Allocated Memory: allocated_by_plan=2566kB allocated_by_exec=31183kB base_allocation=3855kB base_allocation_increase=80kB
```

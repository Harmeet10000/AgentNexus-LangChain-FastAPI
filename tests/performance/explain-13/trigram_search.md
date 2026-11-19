# EXPLAIN (ANALYZE, BUFFERS) — trigram_search

```
Limit  (cost=279.81..279.81 rows=1 width=52) (actual time=6207.159..6207.160 rows=0.00 loops=1)
  Buffers: shared hit=6675
  ->  Sort  (cost=279.81..279.81 rows=1 width=52) (actual time=6207.158..6207.158 rows=0.00 loops=1)
        Sort Key: (similarity(search_text, 'termination obligations compensation'::text)) DESC, id
        Sort Method: quicksort  Memory: 25kB
        Buffers: shared hit=6675
        ->  Bitmap Heap Scan on chunks c  (cost=267.85..279.80 rows=1 width=52) (actual time=6207.144..6207.145 rows=0.00 loops=1)
              Recheck Cond: (search_text % 'termination obligations compensation'::text)
              Rows Removed by Index Recheck: 43956
              Filter: (((user_id)::text = 'scratch-user-00'::text) AND (similarity(search_text, 'termination obligations compensation'::text) >= '0.1'::real))
              Heap Blocks: exact=6375
              Buffers: shared hit=6669
              ->  Bitmap Index Scan on chunks_search_text_trgm_idx  (cost=0.00..267.85 rows=6 width=0) (actual time=25.150..25.150 rows=44556.00 loops=1)
                    Index Cond: (search_text % 'termination obligations compensation'::text)
                    Index Searches: 1
                    Buffers: shared hit=294
Planning:
  Buffers: shared hit=270
Planning Time: 5.324 ms
Execution Time: 6207.317 ms
Allocated Memory: allocated_by_plan=2791kB allocated_by_exec=8971kB base_allocation=4089kB base_allocation_increase=5kB
```

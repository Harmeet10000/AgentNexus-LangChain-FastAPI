# EXPLAIN (ANALYZE, BUFFERS) — legal_rrf_search

```
Limit  (cost=20.81..20.81 rows=1 width=488) (actual time=0.091..0.094 rows=0.00 loops=1)
  Buffers: shared hit=12
  CTE candidate_chunks
    ->  Nested Loop  (cost=0.57..8.70 rows=1 width=735) (actual time=0.021..0.022 rows=0.00 loops=1)
          Buffers: shared hit=3
          ->  Index Scan using pk_chunks on chunks c_1  (cost=0.29..4.32 rows=1 width=735) (actual time=0.020..0.021 rows=0.00 loops=1)
                Index Cond: (id = ANY ('{b8e50222-c791-4805-80fe-f200e10a418f}'::uuid[]))
                Filter: ((document_id = ANY ('{006a88e8-8009-4b65-8270-c4e26a2aab4f}'::uuid[])) AND ((clause_type)::text = 'termination'::text) AND ((metadata_ ->> 'jurisdiction'::text) = 'India'::text) AND ((metadata_ ->> 'contract_type'::text) = 'services'::text))
                Rows Removed by Filter: 1
                Index Searches: 1
                Buffers: shared hit=3
          ->  Index Scan using pk_documents on documents d  (cost=0.28..4.30 rows=1 width=16) (never executed)
                Index Cond: (id = c_1.document_id)
                Filter: ((user_id)::text = 'scratch-user-00'::text)
                Index Searches: 0
  ->  Sort  (cost=12.11..12.12 rows=1 width=488) (actual time=0.090..0.093 rows=0.00 loops=1)
        Sort Key: ((((0.4 * COALESCE((1.0 / (60.0 + (v.rank)::numeric)), 0.0)) + (0.6 * COALESCE((1.0 / (60.0 + ((row_number() OVER w1))::numeric)), 0.0))) + (0.15 * COALESCE((1.0 / (60.0 + (t.rank)::numeric)), 0.0)))) DESC
        Sort Method: quicksort  Memory: 25kB
        Buffers: shared hit=12
        ->  Nested Loop  (cost=5.48..12.10 rows=1 width=488) (actual time=0.074..0.076 rows=0.00 loops=1)
              Buffers: shared hit=9
              ->  Hash Full Join  (cost=5.19..7.75 rows=1 width=72) (actual time=0.074..0.076 rows=0.00 loops=1)
                    Hash Cond: (COALESCE(v.id, candidate_chunks.id) = t.id)
                    Buffers: shared hit=9
                    ->  Hash Full Join  (cost=5.11..7.65 rows=1 width=48) (actual time=0.029..0.030 rows=0.00 loops=1)
                          Hash Cond: (candidate_chunks.id = v.id)
                          Buffers: shared hit=3
                          ->  Limit  (cost=5.03..7.55 rows=1 width=32) (actual time=0.005..0.005 rows=0.00 loops=1)
                                ->  WindowAgg  (cost=5.03..7.55 rows=1 width=32) (actual time=0.005..0.005 rows=0.00 loops=1)
                                      Window: w1 AS (ORDER BY ((candidate_chunks.search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query)) ROWS UNBOUNDED PRECEDING)
                                      ->  Sort  (cost=5.03..5.04 rows=1 width=24) (actual time=0.005..0.005 rows=0.00 loops=1)
                                            Sort Key: ((candidate_chunks.search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query))
                                            Sort Method: quicksort  Memory: 25kB
                                            ->  CTE Scan on candidate_chunks  (cost=0.00..5.02 rows=1 width=24) (actual time=0.000..0.000 rows=0.00 loops=1)
                                                  Filter: ((search_text <@> 'chunks_bm25_idx:termination obligations compensation'::bm25query) < '-1'::double precision)
                                                  Storage: Memory  Maximum Storage: 17kB
                          ->  Hash  (cost=0.07..0.07 rows=1 width=24) (actual time=0.012..0.013 rows=0.00 loops=1)
                                Buckets: 1024  Batches: 1  Memory Usage: 8kB
                                Buffers: shared hit=3
                                ->  Subquery Scan on v  (cost=0.03..0.07 rows=1 width=24) (actual time=0.012..0.012 rows=0.00 loops=1)
                                      Buffers: shared hit=3
                                      ->  Limit  (cost=0.03..0.06 rows=1 width=32) (actual time=0.012..0.012 rows=0.00 loops=1)
                                            Buffers: shared hit=3
                                            ->  WindowAgg  (cost=0.03..0.06 rows=1 width=32) (actual time=0.012..0.012 rows=0.00 loops=1)
                                                  Window: w1 AS (ORDER BY ((candidate_chunks_1.embedding <=> '[<768-float embedding literal>]'::vector)) ROWS UNBOUNDED PRECEDING)
                                                  Buffers: shared hit=3
                                                  ->  Sort  (cost=0.03..0.04 rows=1 width=24) (actual time=0.011..0.012 rows=0.00 loops=1)
                                                        Sort Key: ((candidate_chunks_1.embedding <=> '[<768-float embedding literal>]'::vector))
                                                        Sort Method: quicksort  Memory: 25kB
                                                        Buffers: shared hit=3
                                                        ->  CTE Scan on candidate_chunks candidate_chunks_1  (cost=0.00..0.02 rows=1 width=24) (actual time=0.000..0.000 rows=0.00 loops=1)
                                                              Filter: (embedding IS NOT NULL)
                                                              Storage: Memory  Maximum Storage: 17kB
                    ->  Hash  (cost=0.07..0.07 rows=1 width=24) (actual time=0.036..0.037 rows=0.00 loops=1)
                          Buckets: 1024  Batches: 1  Memory Usage: 8kB
                          Buffers: shared hit=6
                          ->  Subquery Scan on t  (cost=0.04..0.07 rows=1 width=24) (actual time=0.036..0.036 rows=0.00 loops=1)
                                Buffers: shared hit=6
                                ->  Limit  (cost=0.04..0.06 rows=1 width=28) (actual time=0.035..0.036 rows=0.00 loops=1)
                                      Buffers: shared hit=6
                                      ->  WindowAgg  (cost=0.04..0.06 rows=1 width=28) (actual time=0.035..0.035 rows=0.00 loops=1)
                                            Window: w1 AS (ORDER BY (similarity(candidate_chunks_2.search_text, 'termination obligations compensation'::text)) ROWS UNBOUNDED PRECEDING)
                                            Buffers: shared hit=6
                                            ->  Sort  (cost=0.04..0.04 rows=1 width=20) (actual time=0.034..0.034 rows=0.00 loops=1)
                                                  Sort Key: (similarity(candidate_chunks_2.search_text, 'termination obligations compensation'::text)) DESC
                                                  Sort Method: quicksort  Memory: 25kB
                                                  Buffers: shared hit=6
                                                  ->  CTE Scan on candidate_chunks candidate_chunks_2  (cost=0.00..0.02 rows=1 width=20) (actual time=0.022..0.022 rows=0.00 loops=1)
                                                        Filter: (search_text % 'termination obligations compensation'::text)
                                                        Storage: Memory  Maximum Storage: 17kB
                                                        Buffers: shared hit=3
              ->  Index Scan using pk_chunks on chunks c  (cost=0.29..4.31 rows=1 width=424) (never executed)
                    Index Cond: (id = COALESCE(v.id, candidate_chunks.id, t.id))
                    Filter: (search_text ~~* '%termination%'::text)
                    Index Searches: 0
Planning:
  Buffers: shared hit=395 dirtied=1
Planning Time: 6.643 ms
Execution Time: 0.375 ms
Allocated Memory: allocated_by_plan=4963kB allocated_by_exec=1415kB base_allocation=4109kB
```

# Schema versioning (Gesso_Stack.md §LXIX determinism; North Star §34).
#
# Law: internal schemas carry versions. If cache semantics change, bump the
# relevant version. Do not silently reinterpret old records.
#
# Every persisted artifact (receipts, future cache entries, tuning records,
# benchmark result files) stamps the schema version it was written under.

export GESSO_SCHEMA_VERSION, RECEIPT_SCHEMA_VERSION, BENCH_RESULT_SCHEMA_VERSION

"""Overall schema generation for persisted Gesso artifacts."""
const GESSO_SCHEMA_VERSION = v"0.1.0"

"""Receipt record schema (src/receipts.jl). Bump on any field change."""
const RECEIPT_SCHEMA_VERSION = v"0.1.0"

"""Benchmark result file schema (benchmark/results/*.tsv). Bump on format change."""
# v0.1.0: schema,date_utc,julia_version,benchmark,samples,median_ns
#         (rows carry the literal schema tag "bench-result-v1")
# v0.2.0: adds host,arch,nthreads,commit,dirty,bench_note columns and
#         mean_ns,min_ns,allocs,bytes statistics; rows carry the schema
#         version string itself (e.g. "0.2.0"). Result files are append-only
#         (§LXIX): a file may mix row schemas — group by the schema column
#         when comparing; never silently reinterpret old rows.
const BENCH_RESULT_SCHEMA_VERSION = v"0.2.0"

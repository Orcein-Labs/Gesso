# Schema versioning (Harpe_Stack.md §LXIX determinism; North Star §34).
#
# Law: internal schemas carry versions. If cache semantics change, bump the
# relevant version. Do not silently reinterpret old records.
#
# Every persisted artifact (receipts, future cache entries, tuning records,
# benchmark result files) stamps the schema version it was written under.

export HARPE_SCHEMA_VERSION, RECEIPT_SCHEMA_VERSION, BENCH_RESULT_SCHEMA_VERSION

"""Overall schema generation for persisted Harpe artifacts."""
const HARPE_SCHEMA_VERSION = v"0.1.0"

"""Receipt record schema (src/receipts.jl). Bump on any field change."""
const RECEIPT_SCHEMA_VERSION = v"0.1.0"

"""Benchmark result file schema (benchmark/results/*.tsv). Bump on format change."""
const BENCH_RESULT_SCHEMA_VERSION = v"0.1.0"

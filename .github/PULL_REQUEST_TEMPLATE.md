name: Gesso PR
description: Pull request receipt (Gesso_Stack.md §LXXII)
body:
  - type: textarea
    id: what-why
    attributes:
      label: What changed, and why
    validations:
      required: true
  - type: textarea
    id: work-item
    attributes:
      label: Work item
      description: Link the §LXXI work item this PR closes. Refactors beyond it are a violation.
    validations:
      required: true
  - type: textarea
    id: tests
    attributes:
      label: Tests
      description: Which tests pass because of this PR; which were added.
    validations:
      required: true
  - type: textarea
    id: numerical-delta
    attributes:
      label: Numerical delta
      description: Any change in computed results, tolerances, or semantics. "None" must be justified.
    validations:
      required: true
  - type: textarea
    id: benchmarks
    attributes:
      label: Before / after benchmarks
      description: Attach benchmark/results/ files. No benchmark = no performance claims (§LXXII).
  - type: textarea
    id: impacts
    attributes:
      label: Compile-time / memory impact
      description: Specialization counts, load time, allocation notes where relevant.
  - type: textarea
    id: environment
    attributes:
      label: Hardware / workload / model / backend tested
      description: The environment the receipt was earned in.
    validations:
      required: true
  - type: checkboxes
    id: checklist
    attributes:
      label: Charter checklist (AGENTS.md §9)
      options:
        - label: Tests pass (`make test`)
        - label: Formatter run (`make format`)
        - label: Receipt complete (this PR is the receipt)
        - label: Changed files ⊆ permitted files
        - label: Known limitations and unresolved questions written down
        - label: No speculative future implementation

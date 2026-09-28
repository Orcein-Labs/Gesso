name: Work Item
description: A bounded, testable swarm task (Harpe_Stack.md §LXXI format)
labels: ["work-item"]
body:
  - type: textarea
    id: objective
    attributes:
      label: Objective
      description: One sentence. What outcome, not what steps.
    validations:
      required: true
  - type: textarea
    id: permitted-files
    attributes:
      label: Permitted files
      description: Exact paths this item may create or modify. Anything else requires a new item.
    validations:
      required: true
  - type: textarea
    id: interfaces
    attributes:
      label: Interfaces
      description: Types/functions this item consumes or must expose. Cite governing doc sections.
    validations:
      required: true
  - type: textarea
    id: invariants
    attributes:
      label: Invariants
      description: Properties that must hold after this item lands (and stay holding).
    validations:
      required: true
  - type: textarea
    id: tests
    attributes:
      label: Tests
      description: Which tests prove the objective. New tests this item must add.
    validations:
      required: true
  - type: textarea
    id: performance-target
    attributes:
      label: Performance target
      description: If relevant — the measurable bar and the benchmark that measures it. Else "N/A".
  - type: textarea
    id: expected-artifact
    attributes:
      label: Expected artifact
      description: What exists after this item that did not exist before (code, doc, receipt, result file).
    validations:
      required: true
  - type: textarea
    id: receipt
    attributes:
      label: Receipt (filled at close — §LXXII)
      description: >
        what changed · why · tests · numerical delta · before benchmark ·
        after benchmark · compile-time impact · memory impact · hardware ·
        workload · model · backend

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    wall_clock_budget_seconds: float = 600.0
    population_size: int = 10
    offspring_size: int = 10
    tournament_size: int = 3
    crossover_probability: float = 0.7
    mutation_probability: float = 0.9
    max_preprocessing_nodes: int = 8
    max_depth: int = 6
    max_branches: int = 4
    stage1_subsample_fraction: float = 0.3
    stage1_tolerance: float = 0.02
    novelty_threshold: float = 0.5
    cost_alpha: float = 0.01
    cost_beta: float = 0.001
    cv_folds: int = 5
    screening_test_size: float = 0.3
    random_state: int = 42
    n_jobs: int = 1
    per_eval_timeout_seconds: float | None = None
    # Survival-selection penalties used by WorkflowIndividual.fitness().
    # The reported incumbent is still selected by raw full_score. Setting both
    # to 0 removes the possible confound that the EA survival objective
    # penalizes the richer structures being evaluated.
    fitness_cost_penalty: float = 1e-4
    fitness_complexity_penalty: float = 1e-3

@dataclass
class MethodConfig:
    name: str = "proposed"
    allow_branching: bool = True
    use_parallel_cost: bool = True
    use_staged_evaluation: bool = True
    use_evolutionary_search: bool = True
    # --- benchmark axes (added) ---
    # structure axis: "linear" | "branch" | "dag".
    # If left None, it is inferred from allow_branching so the original
    # 5 presets keep their exact behaviour.
    structure: str | None = None
    # budget axis: when True, candidates whose predicted completion time
    # exceeds the remaining wall-clock budget are pruned before evaluation.
    use_budget_aware: bool = False
    # derived in __post_init__ -- do NOT set directly:
    allow_branch_mutation: bool = True

    def __post_init__(self) -> None:
        if self.structure is None:
            self.structure = "dag" if self.allow_branching else "linear"
        # structure is the source of truth once resolved
        self.allow_branching = self.structure in ("branch", "dag")
        # only the free DAG level may change the branch topology by mutation
        self.allow_branch_mutation = self.structure == "dag"

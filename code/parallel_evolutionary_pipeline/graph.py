from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import random

from .config import MethodConfig, SearchConfig
from .datasets import OpenMLDataset
from .operators import BRANCH_OPERATORS, ESTIMATOR_OPERATORS, TAIL_OPERATORS


@dataclass
class OperatorGene:
    operator: str
    params: dict[str, Any]


@dataclass
class BranchGene:
    scope: str
    nodes: list[OperatorGene] = field(default_factory=list)


@dataclass
class WorkflowGraph:
    branches: list[BranchGene]
    tail_nodes: list[OperatorGene]
    estimator: OperatorGene

    def clone(self) -> "WorkflowGraph":
        return copy.deepcopy(self)

    def preprocessing_nodes(self) -> int:
        return sum(len(branch.nodes) for branch in self.branches) + len(self.tail_nodes)

    def depth(self) -> int:
        branch_depth = max((len(branch.nodes) for branch in self.branches), default=0)
        return branch_depth + len(self.tail_nodes) + 2

    def branch_count(self) -> int:
        return len(self.branches)

    def complexity(self) -> int:
        return self.preprocessing_nodes() + self.branch_count()


@dataclass
class WorkflowIndividual:
    graph: WorkflowGraph
    estimated_cost: float | None = None
    serial_cost: float | None = None
    stage1_score: float | None = None
    full_score: float | None = None
    runtime_seconds: float | None = None
    status: str = "unevaluated"
    novelty: float = 0.0

    # Penalty coefficients applied ONLY in the EA's survival selection.
    # The reported incumbent is tracked by raw balanced accuracy (full_score),
    # so setting both to 0.0 makes survival use the same objective the
    # configurations are finally compared on (removes the objective mismatch
    # vs. the random controls). Plain class attrs (no annotation) so they are
    # not dataclass fields and can be toggled globally before a run.
    COST_PENALTY = 1e-4
    COMPLEXITY_PENALTY = 1e-3

    def fitness(self) -> float:
        score = self.full_score if self.full_score is not None else (self.stage1_score or 0.0)
        return score - self.COST_PENALTY * (self.estimated_cost or 0.0) - self.COMPLEXITY_PENALTY * self.graph.complexity()


def random_gene(pool: dict[str, Any], rng: random.Random, random_state: int = 0) -> OperatorGene:
    name = rng.choice(list(pool.keys()))
    spec = pool[name]
    params = dict(spec.default_params)
    for key, values in spec.param_space.items():
        params[key] = rng.choice(values)
    if "random_state" in params:
        params["random_state"] = random_state
    return OperatorGene(operator=name, params=params)


def random_compatible_branch_gene(signature: str, rng: random.Random, random_state: int = 0) -> OperatorGene:
    compatible = {name: spec for name, spec in BRANCH_OPERATORS.items() if signature in spec.allowed_inputs}
    return random_gene(compatible, rng, random_state)


def random_tail_gene(rng: random.Random, random_state: int = 0) -> OperatorGene:
    return random_gene(TAIL_OPERATORS, rng, random_state)


def _scope_candidates(dataset: OpenMLDataset, method: MethodConfig) -> list[str]:
    scopes: list[str] = []
    if dataset.numeric_columns:
        scopes.append("numeric")
    if dataset.categorical_columns:
        scopes.append("categorical")
    if method.allow_branching and dataset.numeric_columns and dataset.categorical_columns:
        scopes.append("all")
    if not scopes:
        scopes.append("all")
    return scopes


def make_fixed_branch_graph(dataset: OpenMLDataset, config: SearchConfig, method: MethodConfig, rng: random.Random) -> WorkflowGraph:
    """Fixed numeric/categorical two-branch template (structure='branch').

    Numeric columns flow through a numeric preprocessing branch; categorical
    columns flow through an (optional imputer +) encoder branch that outputs
    numeric features. The topology is fixed: branch-count-changing mutations
    are disabled for this structure level (allow_branch_mutation=False).
    """
    branches: list[BranchGene] = []
    if dataset.numeric_columns:
        nodes = [random_compatible_branch_gene("numeric", rng, config.random_state)]
        branches.append(BranchGene(scope="numeric", nodes=nodes))
    if dataset.categorical_columns:
        nodes: list[OperatorGene] = []
        if rng.random() < 0.5:
            nodes.append(OperatorGene(operator="simple_imputer", params={"strategy": "most_frequent"}))
        enc_pool = {n: BRANCH_OPERATORS[n] for n in ("one_hot_encoder", "ordinal_encoder", "target_encoder") if n in BRANCH_OPERATORS}
        nodes.append(random_gene(enc_pool, rng, config.random_state))
        branches.append(BranchGene(scope="categorical", nodes=nodes))
    if not branches:
        branches.append(BranchGene(scope="all", nodes=[random_compatible_branch_gene("mixed", rng, config.random_state)]))
    tail_nodes: list[OperatorGene] = []
    if rng.random() < 0.5:
        tail_nodes.append(random_tail_gene(rng, config.random_state))
    estimator = random_gene(ESTIMATOR_OPERATORS, rng, config.random_state)
    return WorkflowGraph(branches=branches, tail_nodes=tail_nodes, estimator=estimator)


def make_random_graph(dataset: OpenMLDataset, config: SearchConfig, method: MethodConfig, rng: random.Random) -> WorkflowGraph:
    if getattr(method, "structure", None) == "branch":
        return make_fixed_branch_graph(dataset, config, method, rng)
    scopes = _scope_candidates(dataset, method)
    if method.allow_branching:
        n_branches = rng.randint(1, min(config.max_branches, len(scopes)))
    else:
        n_branches = 1
    chosen_scopes = scopes[:]
    rng.shuffle(chosen_scopes)
    branches: list[BranchGene] = []
    for i in range(n_branches):
        scope = chosen_scopes[i % len(chosen_scopes)]
        depth = rng.randint(1, max(1, config.max_depth - 2))
        nodes: list[OperatorGene] = []
        signature = "numeric" if scope == "numeric" else ("categorical" if scope == "categorical" else "mixed")
        for _ in range(min(depth, 2)):
            gene = random_compatible_branch_gene(signature, rng, config.random_state)
            nodes.append(gene)
            spec = BRANCH_OPERATORS[gene.operator]
            signature = signature if spec.output_signature == "same" else spec.output_signature
            if signature == "numeric" and rng.random() < 0.6:
                break
        if signature != "numeric":
            nodes.append(random_compatible_branch_gene(signature, rng, config.random_state))
        branches.append(BranchGene(scope=scope, nodes=nodes))
    tail_nodes: list[OperatorGene] = []
    if rng.random() < 0.6:
        tail_nodes.append(random_tail_gene(rng, config.random_state))
    estimator = random_gene(ESTIMATOR_OPERATORS, rng, config.random_state)
    return WorkflowGraph(branches=branches, tail_nodes=tail_nodes, estimator=estimator)


def is_feasible(graph: WorkflowGraph, dataset: OpenMLDataset, config: SearchConfig, method: MethodConfig) -> bool:
    if not graph.branches:
        return False
    if not method.allow_branching and len(graph.branches) != 1:
        return False
    if graph.preprocessing_nodes() > config.max_preprocessing_nodes:
        return False
    if graph.depth() > config.max_depth:
        return False
    if graph.branch_count() > config.max_branches:
        return False
    seen_smote = 0
    for branch in graph.branches:
        if branch.scope == "numeric" and not dataset.numeric_columns:
            return False
        if branch.scope == "categorical" and not dataset.categorical_columns:
            return False
        signature = "numeric" if branch.scope == "numeric" else ("categorical" if branch.scope == "categorical" else "mixed")
        for node in branch.nodes:
            spec = BRANCH_OPERATORS.get(node.operator)
            if spec is None or signature not in spec.allowed_inputs:
                return False
            signature = signature if spec.output_signature == "same" else spec.output_signature
        if signature != "numeric":
            return False
    tail_signature = "numeric"
    for node in graph.tail_nodes:
        spec = TAIL_OPERATORS.get(node.operator)
        if spec is None or tail_signature not in spec.allowed_inputs:
            return False
        if node.operator == "smote":
            seen_smote += 1
        tail_signature = spec.output_signature
    if seen_smote > 1:
        return False
    return graph.estimator.operator in ESTIMATOR_OPERATORS


def initialize_population(dataset: OpenMLDataset, config: SearchConfig, method: MethodConfig) -> list[WorkflowIndividual]:
    rng = random.Random(config.random_state)
    population: list[WorkflowIndividual] = []
    while len(population) < config.population_size:
        graph = make_random_graph(dataset, config, method, rng)
        if is_feasible(graph, dataset, config, method):
            population.append(WorkflowIndividual(graph=graph))
    return population

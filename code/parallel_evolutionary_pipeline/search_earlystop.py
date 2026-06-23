from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .config import MethodConfig, SearchConfig
from .datasets import OpenMLDataset
from .evaluation_earlystop import estimate_cost, evaluate_graph_cv, evaluate_graph_screen
from .graph import (
    BranchGene,
    OperatorGene,
    WorkflowGraph,
    WorkflowIndividual,
    initialize_population,
    is_feasible,
    make_random_graph,
    random_gene,
)
from .operators import BRANCH_OPERATORS, ESTIMATOR_OPERATORS, TAIL_OPERATORS


@dataclass
class SearchResult:
    best_individual: WorkflowIndividual
    history: list[dict]
    pruned: int = 0


METHOD_PRESETS = {
    "proposed": MethodConfig(name="proposed", allow_branching=True, use_parallel_cost=True, use_staged_evaluation=True, use_evolutionary_search=True),
    "linear_ea": MethodConfig(name="linear_ea", allow_branching=False, use_parallel_cost=False, use_staged_evaluation=True, use_evolutionary_search=True),
    "graph_serialcost": MethodConfig(name="graph_serialcost", allow_branching=True, use_parallel_cost=False, use_staged_evaluation=True, use_evolutionary_search=True),
    "graph_nostaged": MethodConfig(name="graph_nostaged", allow_branching=True, use_parallel_cost=True, use_staged_evaluation=False, use_evolutionary_search=True),
    "random_graph": MethodConfig(name="random_graph", allow_branching=True, use_parallel_cost=True, use_staged_evaluation=True, use_evolutionary_search=False),
}


# ID = structure - cost - budget - eval :
#   structure: L=linear, B=fixed branch, D=typed DAG
#   cost:      S=serial-sum, C=critical-path
#   budget:    A=agnostic, B=budget-aware
#   eval:      F=full, T=staged
RUN_MATRIX = {
    "L-S-F-A": MethodConfig(name="L-S-F-A", structure="linear", use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=False),
    "B-S-F-A": MethodConfig(name="B-S-F-A", structure="branch", use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=False),
    "D-S-F-A": MethodConfig(name="D-S-F-A", structure="dag",    use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=False),
    "D-C-F-A": MethodConfig(name="D-C-F-A", structure="dag",    use_parallel_cost=True,  use_budget_aware=False, use_staged_evaluation=False),
    "D-S-F-B": MethodConfig(name="D-S-F-B", structure="dag",    use_parallel_cost=False, use_budget_aware=True,  use_staged_evaluation=False),
    "D-S-T-A": MethodConfig(name="D-S-T-A", structure="dag",    use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=True),
    "D-C-T-B": MethodConfig(name="D-C-T-B", structure="dag",    use_parallel_cost=True,  use_budget_aware=True,  use_staged_evaluation=True),
    "D-Rand":  MethodConfig(name="D-Rand",  structure="dag",    use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=False, use_evolutionary_search=False),
    "Flat-Rand": MethodConfig(name="Flat-Rand", structure="linear", use_parallel_cost=False, use_budget_aware=False, use_staged_evaluation=False, use_evolutionary_search=False),
}


class ParallelAwareEA:
    def __init__(self, config: SearchConfig | None = None, method: MethodConfig | None = None):
        self.config = config or SearchConfig()
        self.method = method or METHOD_PRESETS["proposed"]
        self.rng = random.Random(self.config.random_state)
        # Elitism: guarantee the best-accuracy incumbent stays in the breeding
        # pool. Default ON; flip to False to reproduce the original behaviour.
        self.use_elitism = True

    def _graph_signature(self, graph: WorkflowGraph) -> tuple:
        branches = tuple(sorted((branch.scope, tuple(node.operator for node in branch.nodes)) for branch in graph.branches))
        tail = tuple(node.operator for node in graph.tail_nodes)
        return branches + (tail, graph.estimator.operator)

    def _novelty(self, graph: WorkflowGraph, seen: list[WorkflowIndividual]) -> float:
        sig = self._graph_signature(graph)
        existing = {self._graph_signature(ind.graph) for ind in seen}
        return 1.0 if sig not in existing else 0.0

    def _set_cost_fields(self, individual: WorkflowIndividual, screen_score: float | None, total_cost: float, serial_cost: float) -> None:
        individual.stage1_score = screen_score
        individual.estimated_cost = total_cost if self.method.use_parallel_cost else serial_cost
        individual.serial_cost = serial_cost

    def _evaluate_individual(self, individual: WorkflowIndividual, dataset: OpenMLDataset, incumbent: float | None) -> None:
        timeout = self.config.per_eval_timeout_seconds
        if self.method.use_staged_evaluation:
            screen = evaluate_graph_screen(individual.graph, dataset, self.config, hard_timeout=timeout)
            self._set_cost_fields(individual, screen.score, screen.cost.total_cost, screen.cost.serial_cost)
            promote = incumbent is None or screen.score >= incumbent - self.config.stage1_tolerance
            promote = promote or individual.novelty >= self.config.novelty_threshold
            if promote:
                full = evaluate_graph_cv(individual.graph, dataset, self.config, hard_timeout=timeout)
                individual.full_score = full.score
                individual.runtime_seconds = full.runtime_seconds
                individual.status = full.status
            else:
                individual.status = "screened_out"
        else:
            full = evaluate_graph_cv(individual.graph, dataset, self.config, hard_timeout=timeout)
            self._set_cost_fields(individual, full.score, full.cost.total_cost, full.cost.serial_cost)
            individual.full_score = full.score
            individual.runtime_seconds = full.runtime_seconds
            individual.status = full.status

    def _select_parent(self, population: list[WorkflowIndividual]) -> WorkflowIndividual:
        sample = self.rng.sample(population, k=min(self.config.tournament_size, len(population)))
        return max(sample, key=lambda ind: ind.fitness())

    def _mutate_add_node(self, graph: WorkflowGraph) -> str:
        branch = self.rng.choice(graph.branches)
        if graph.preprocessing_nodes() >= self.config.max_preprocessing_nodes:
            return "add_node"
        branch.nodes.insert(self.rng.randint(0, len(branch.nodes)), random_gene(BRANCH_OPERATORS, self.rng, self.config.random_state))
        return "add_node"

    def _mutate_remove_node(self, graph: WorkflowGraph) -> str:
        candidates = [branch for branch in graph.branches if branch.nodes]
        if not candidates:
            return "remove_node"
        branch = self.rng.choice(candidates)
        del branch.nodes[self.rng.randrange(len(branch.nodes))]
        return "remove_node"

    def _mutate_replace_op(self, graph: WorkflowGraph) -> str:
        if self.rng.random() < 0.25 and graph.tail_nodes:
            idx = self.rng.randrange(len(graph.tail_nodes))
            graph.tail_nodes[idx] = random_gene(TAIL_OPERATORS, self.rng, self.config.random_state)
            return "replace_op"
        branch = self.rng.choice(graph.branches)
        if branch.nodes:
            branch.nodes[self.rng.randrange(len(branch.nodes))] = random_gene(BRANCH_OPERATORS, self.rng, self.config.random_state)
        else:
            branch.nodes.append(random_gene(BRANCH_OPERATORS, self.rng, self.config.random_state))
        return "replace_op"

    def _mutate_branch_split(self, graph: WorkflowGraph, dataset: OpenMLDataset) -> str:
        if not self.method.allow_branch_mutation or len(graph.branches) >= self.config.max_branches:
            return "branch_split"
        source = self.rng.choice(graph.branches)
        prefix = source.nodes[: self.rng.randint(0, min(2, len(source.nodes)))]
        graph.branches.append(BranchGene(scope=source.scope, nodes=list(prefix)))
        if not graph.branches[-1].nodes:
            graph.branches[-1].nodes.append(random_gene(BRANCH_OPERATORS, self.rng, self.config.random_state))
        return "branch_split"

    def _mutate_hyper(self, graph: WorkflowGraph) -> str:
        if self.rng.random() < 0.25:
            spec = ESTIMATOR_OPERATORS[graph.estimator.operator]
            for key, values in spec.param_space.items():
                if self.rng.random() < 0.5:
                    graph.estimator.params[key] = self.rng.choice(values)
            return "hyper_mutate"
        if graph.tail_nodes and self.rng.random() < 0.3:
            gene = self.rng.choice(graph.tail_nodes)
            spec = TAIL_OPERATORS[gene.operator]
        else:
            branch = self.rng.choice(graph.branches)
            if not branch.nodes:
                return "hyper_mutate"
            gene = self.rng.choice(branch.nodes)
            spec = BRANCH_OPERATORS[gene.operator]
        for key, values in spec.param_space.items():
            if self.rng.random() < 0.5:
                gene.params[key] = self.rng.choice(values)
        return "hyper_mutate"

    def _mutate(self, parent: WorkflowIndividual, dataset: OpenMLDataset) -> tuple[WorkflowIndividual, str]:
        graph = parent.graph.clone()
        ops = [self._mutate_add_node, self._mutate_remove_node, self._mutate_replace_op, self._mutate_hyper]
        if self.method.allow_branch_mutation:
            ops.append(lambda g: self._mutate_branch_split(g, dataset))
        op_name = self.rng.choice(ops)(graph)
        if not is_feasible(graph, dataset, self.config, self.method):
            return WorkflowIndividual(parent.graph.clone()), op_name
        return WorkflowIndividual(graph=graph), op_name

    def _crossover(self, a: WorkflowIndividual, b: WorkflowIndividual, dataset: OpenMLDataset) -> tuple[WorkflowIndividual, str]:
        graph = a.graph.clone()
        donor = b.graph.clone()
        if self.method.allow_branch_mutation and graph.branches and donor.branches:
            graph.branches[self.rng.randrange(len(graph.branches))] = self.rng.choice(donor.branches)
        if self.rng.random() < 0.3:
            graph.tail_nodes = donor.tail_nodes[:1]
        if self.rng.random() < 0.3:
            graph.estimator = donor.estimator
        if not is_feasible(graph, dataset, self.config, self.method):
            return WorkflowIndividual(a.graph.clone()), "branch_crossover"
        return WorkflowIndividual(graph=graph), "branch_crossover"

    def _run_random_search(self, dataset: OpenMLDataset) -> SearchResult:
        start = time.perf_counter()
        history: list[dict] = []
        best = WorkflowIndividual(make_random_graph(dataset, self.config, self.method, self.rng))
        best.full_score = -1.0
        while time.perf_counter() - start < self.config.wall_clock_budget_seconds:
            cand = WorkflowIndividual(make_random_graph(dataset, self.config, self.method, self.rng))
            if not is_feasible(cand.graph, dataset, self.config, self.method):
                continue
            cand.novelty = 1.0
            self._evaluate_individual(cand, dataset, best.full_score if best.full_score is not None and best.full_score >= 0 else None)
            if cand.full_score is not None and (best.full_score is None or cand.full_score > best.full_score):
                best = cand
            history.append({"status": cand.status, "score": cand.full_score, "best_so_far": best.full_score, "elapsed_seconds": time.perf_counter() - start, "generation": None, "operator": "random_graph", "branches": cand.graph.branch_count(), "nodes": cand.graph.preprocessing_nodes()})
        return SearchResult(best_individual=best, history=history)

    def fit(self, dataset: OpenMLDataset) -> SearchResult:
        if not self.method.use_evolutionary_search:
            return self._run_random_search(dataset)
        start = time.perf_counter()
        history: list[dict] = []
        eval_cost_pairs: list[tuple[float, float]] = []  # (estimated_cost, measured runtime) for budget calibration
        n_pruned = 0
        population = initialize_population(dataset, self.config, self.method)
        incumbent: WorkflowIndividual | None = None
        # Guarantee at least this many evaluated individuals so the parent pool has
        # enough diversity; evaluated after the minimum are subject to budget checks.
        min_initial = min(3, len(population))
        for i, individual in enumerate(population):
            elapsed = time.perf_counter() - start
            remaining = self.config.wall_clock_budget_seconds - elapsed
            # Budget completely exhausted — stop entirely.
            if remaining <= 0:
                break
            # After the minimum guaranteed batch, skip individuals whose evaluation
            # is unlikely to finish within the remaining budget.
            if i >= min_initial and self.config.per_eval_timeout_seconds and remaining < self.config.per_eval_timeout_seconds:
                continue
            individual.novelty = 1.0
            self._evaluate_individual(individual, dataset, None if incumbent is None else incumbent.full_score)
            if individual.status == "ok" and individual.runtime_seconds and individual.estimated_cost and individual.estimated_cost > 0:
                eval_cost_pairs.append((individual.estimated_cost, individual.runtime_seconds))
            if individual.full_score is not None and (incumbent is None or individual.full_score > incumbent.full_score):
                incumbent = individual
        if incumbent is None:
            incumbent = population[0]

        generation = 0
        while time.perf_counter() - start < self.config.wall_clock_budget_seconds:
            offspring: list[WorkflowIndividual] = []
            while len(offspring) < self.config.offspring_size and time.perf_counter() - start < self.config.wall_clock_budget_seconds:
                parent_a = self._select_parent(population)
                if self.rng.random() < self.config.mutation_probability:
                    child, operator_name = self._mutate(parent_a, dataset)
                else:
                    child, operator_name = WorkflowIndividual(parent_a.graph.clone()), "clone"
                if self.rng.random() < self.config.crossover_probability:
                    child, operator_name = self._crossover(child, self._select_parent(population), dataset)
                if is_feasible(child.graph, dataset, self.config, self.method):
                    child.novelty = self._novelty(child.graph, population + offspring)
                    previous_best = incumbent.full_score or -1.0
                    # --- budget axis: prune candidates predicted not to finish in the remaining budget ---
                    if self.method.use_budget_aware and len(eval_cost_pairs) >= 3:
                        cost = estimate_cost(child.graph, dataset, self.config)
                        c_pred = cost.total_cost if self.method.use_parallel_cost else cost.serial_cost
                        t_remain = self.config.wall_clock_budget_seconds - (time.perf_counter() - start)
                        ratios = sorted(rt / ec for ec, rt in eval_cost_pairs if ec > 0)
                        if ratios and c_pred > 0:
                            k = ratios[len(ratios) // 2]  # median seconds-per-cost-unit (calibrated online)
                            if k * c_pred > max(0.0, t_remain):
                                n_pruned += 1
                                continue
                    self._evaluate_individual(child, dataset, incumbent.full_score)
                    if child.status == "ok" and child.runtime_seconds and child.estimated_cost and child.estimated_cost > 0:
                        eval_cost_pairs.append((child.estimated_cost, child.runtime_seconds))
                    offspring.append(child)
                    improved = False
                    if child.full_score is not None and child.status != "screened_out" and child.full_score > previous_best:
                        incumbent = child
                        improved = True
                    history.append(
                        {
                            "status": child.status,
                            "score": child.full_score,
                            "stage1_score": child.stage1_score,
                            "best_so_far": incumbent.full_score,
                            "improved_incumbent": improved,
                            "branches": child.graph.branch_count(),
                            "nodes": child.graph.preprocessing_nodes(),
                            "operator": operator_name,
                            "generation": generation,
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    )
            population = sorted(population + offspring, key=lambda ind: ind.fitness(), reverse=True)[: self.config.population_size]
            # Elitism: never drop the best-accuracy individual from the pool.
            if self.use_elitism and incumbent is not None and not any(ind is incumbent for ind in population):
                population[-1] = incumbent
            generation += 1
        return SearchResult(best_individual=incumbent, history=history, pruned=n_pruned)

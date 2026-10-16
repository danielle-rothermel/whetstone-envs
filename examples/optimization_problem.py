from whetstone_envs.evaluation import (
    ExactMatchEvalProcedureRunner,
    ExperimentContract,
    prepare_experiment,
)
from whetstone_envs.instances import Instance, InstanceGroup, InstancePool
from whetstone_envs.optim.problem import OptimizationProblem

pool = InstancePool(
    instance_group=InstanceGroup(
        name="echo",
        instances=tuple(
            Instance(
                id=f"echo-{index}",
                seed=index,
                categories=("echo",),
                prompt_inputs={"input": f"item-{index}"},
                gold=f"item-{index}",
            )
            for index in range(8)
        ),
    )
)
contract = ExperimentContract(
    namespace="example.echo",
    dataset_revision="echo/v1",
    mutation_field="instruction",
    prompt_fields=("input",),
    response_field="response",
    root_base_schema="example.echo.root_candidate",
    reward_policy_name="echo-exact-match",
    candidate_id_prefix="echo",
    naive_template="Repeat this: {input}",
    ceiling_template="Return only: {input}",
)
problem = OptimizationProblem(
    prepared=prepare_experiment(
        pool, contract=contract, split_sizes=(4, 2, 2)
    ),
    contract=contract,
    eval_runner=ExactMatchEvalProcedureRunner(),
    task_context="Repeat the input exactly.",
    rendering_rules="Substitute the input into the prompt template.",
    example_execution="Compare the reply to the input using exact match.",
    proposal_bodies=(contract.ceiling_template, contract.naive_template),
    fake_replies={
        f"Return only: item-{index}": f"item-{index}" for index in range(8)
    },
)

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

# from whetstone_envs.c22 import PROBES, generate_pool, score
from whetstone_envs.c22 import PROBES, generate_pool

# from whetstone_envs.c22.constraints import ConstraintStack
# from whetstone_envs.c22.oracle import evaluate


def main(console: Console, num_examples: int):
    console.print(Text("C22 | IFEval Variant", style="bold"))
    instances = generate_pool().instances
    for i, instance in enumerate(instances[:num_examples]):
        console.print(
            Panel(
                PROBES.render_naive(instance),
                title=f"Instance #{i + 1}: {instance.id}",
            )
        )
        console.print(Pretty(instance))


if __name__ == "__main__":
    console = Console()
    num_examples = 1
    main(console, num_examples=num_examples)

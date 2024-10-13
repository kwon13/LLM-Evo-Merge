# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import os
import time
from typing import List, Optional

import click
import cma
import numpy as np
import pandas
import ray
import torch
import tqdm
import transformers
import yaml
from functools import reduce
import operator

try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference
from mergekit.llm_evo.config import (
    EvolMergeConfiguration,
    ModelGenomeDefinition,
    check_for_naughty_config,
)
from mergekit.llm_evo.llm_prompts import system_prompt_template, user_prompt_template
from mergekit.llm_evo.genome import ModelGenome
from mergekit.llm_evo.strategy import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
    LLMEvolutionStrategy,
)
from mergekit.merge import run_merge
from mergekit.options import MergeOptions


@click.command("mergekit-llm_evolve")
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=10)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    help="Evaluation scheduling strategy",
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    default=False,
    help="Use in-memory merge & evaluation",
)
@click.option(
    "--storage-path",
    type=str,
    help="Path to storage accessible to all nodes for model storage",
    required=True,
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option("--merge-cuda/--no-merge-cuda", is_flag=True, default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option("--allow-crimes/--no-allow-crimes", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=0)
@click.option("--batch-size", type=int, default=None, help="Batch size for evaluation")
@click.option("--sigma-low", type=float, default=0.05, help="Initial sigma_low for LLMEvo")
@click.option("--sigma-high", type=float, default=0.2, help="Initial sigma_high for LLMEvo")
@click.option("--max-retries", type=int, default=3, help="Initial max_retries for LLMEvo")
@click.option("--api-key", type=str, default=None, help="OpenAI API Key")
@click.option("--llm-model", type=str, default="gpt-4o", help="OpenAI Model Type")
@click.option("use_wandb", "--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--wandb-project", type=str, help="Wandb project name")
@click.option("--wandb-entity", type=str, help="Wandb entity name")
@click.option(
    "--task-search-path",
    type=str,
    multiple=True,
    help="Path to search for lmeval tasks",
)
@click.option(
    "--i-understand-the-depths-of-the-evils-i-am-unleashing",
    "allow_benchmark_tasks",
    is_flag=True,
    default=False,
    help="Allow benchmark tasks as objectives",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    is_flag=True,
    default=True,
    help="Save the final merged model",
)
@click.option(
    "--reshard/--no-reshard",
    is_flag=True,
    default=True,
    help="Convert models to single-shard safetensors for faster merge",
)
@click.option(
    "--force-population-size",
    type=int,
    default=None,
    help="Force a specific initial population size for LLMEvo",
)
def main(
    genome_config_path: str,
    max_fevals: int,
    vllm: bool,
    strategy: str,
    in_memory: bool,
    storage_path: Optional[str],
    num_gpus: Optional[int],
    merge_cuda: bool,
    trust_remote_code: bool,
    allow_crimes: bool,
    random_seed: int,
    batch_size: Optional[int],
    sigma_low: Optional[float],
    sigma_high: Optional[float],
    max_retries: Optional[float],
    api_key: str,
    llm_model:str,
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    task_search_path: List[str],
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    reshard: bool,
    force_population_size: Optional[int],
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )

    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    if use_wandb:
        if not wandb:
            raise RuntimeError("wandb is not installed")
        run = wandb.init(
            project=wandb_project or "mergekit-evolve",
            entity=wandb_entity,
            config=config.model_dump(mode="json"),
        )
    else:
        run = None

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,  # one trillion bytes!
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    # convert models to single-shard safetensors
    if reshard:
        resharded_models = []
        resharded_base = None
        for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):
            resharded_models.append(
                _reshard_model(
                    model,
                    storage_path,
                    merge_options.lora_merge_cache,
                    trust_remote_code,
                )
            )
        if config.genome.base_model is not None:
            resharded_base = _reshard_model(
                config.genome.base_model,
                storage_path,
                merge_options.lora_merge_cache,
                trust_remote_code,
            )
    else:
        resharded_models = config.genome.models
        resharded_base = config.genome.base_model

    genome = ModelGenome(
        ModelGenomeDefinition.model_validate(
            {
                **config.genome.model_dump(
                    exclude=[
                        "models",
                        "base_model",
                    ]
                ),
                "models": resharded_models,
                "base_model": resharded_base,
            }
        ),
        trust_remote_code=trust_remote_code,
    )
    dimensions = genome.dimensions
    
    if strategy == "pool":
        strat_cls = ActorPoolEvaluationStrategy
    elif strategy == "buffered":
        strat_cls = BufferedRayEvaluationStrategy
    elif strategy == "serial":
        strat_cls = SerialEvaluationStrategy
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    strat = strat_cls(
        config,
        genome,
        merge_options,
        num_gpus=num_gpus,
        vllm=vllm,
        in_memory=in_memory,
        model_storage_path=os.path.join(storage_path, "merged"),
        batch_size=batch_size,
        task_search_path=task_search_path,
    )
    
    if force_population_size is not None:
        population_size = force_population_size
    else: population_size = 4 + round(3*np.log(reduce(operator.mul, dimensions, 1)))
    
    x_t = genome.initial_genotype(population_size).view(population_size, -1).numpy()
    xbest=x_t[0]
    xbest_cost = -np.inf

    def progress_callback(es: LLMEvolutionStrategy, evaluations:int):
        nonlocal xbest, xbest_cost

        if use_wandb:
            best_params = genome.genotype_to_param_arrays(es.prev_best_genome)
            run.log(
                {
                    "best_score": es.prev_best_cost,
                    "best_genome": wandb.Table(data=pandas.DataFrame(best_params)),
                    "thought": es.thought,
                    "evaluations": evaluations,
                },
                commit=True,
                step=evaluations,
            )

        if es.prev_best_cost > xbest_cost:
            xbest_cost = es.prev_best_cost
            xbest = es.prev_best_genome
            print(f"New best score: {xbest_cost:.4f}")
            best_yaml = genome.genotype_merge_config(xbest).to_yaml()
            with open(os.path.join(storage_path, "best_config.yaml"), "w") as f:
                f.write(best_yaml)
            print(f"Merge configuration:\n{best_yaml}")

            if use_wandb:
                art = wandb.Artifact("best_config", type="merge_config")
                art.add_file(os.path.join(storage_path, "best_config.yaml"))
                run.log_artifact(art)

    def parallel_evaluate(genomes: List[np.ndarray]) -> List[float]:
        print(f"Received {len(genomes)} genotypes")
        res = strat.evaluate_genotypes(genomes)

        if use_wandb:
            res = list(res)
            score_mean = np.mean([r["score"] for r in res])
            score_std = np.std([r["score"] for r in res])
            run.log(
                {
                    "population/score_mean": score_mean,
                    "population/score_std": score_std,
                },
                commit=False,
            )
            for task in res[0]["results"]:
                for metric in res[0]["results"][task]:
                    values = [r["results"][task][metric] for r in res]
                    values = [v for v in values if v is not None]
                    if not values or all(isinstance(v, str) for v in values):
                        continue

                    mean = np.mean(values)
                    max_val = max(values)
                    min_val = min(values)

                    metric_pretty = metric.replace(",none", "")
                    if metric_pretty.endswith("_stderr"):
                        # don't log stats for stderr that's just silly
                        continue

                    run.log(
                        {
                            f"population/{task}_{metric_pretty}_mean": mean,
                            f"population/{task}_{metric_pretty}_max": max_val,
                            f"population/{task}_{metric_pretty}_min": min_val,
                        },
                        commit=False,
                    )
        return (genomes, [x["score"] for x in res])

    try:
        llm_evo = LLMEvolutionStrategy(
            api_key=api_key,
            model=llm_model,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            dimensions=dimensions,
            population_size=population_size,
        )
        
        for idx in range(max_fevals):
            x_t = llm_evo.mutate(*parallel_evaluate(x_t), sigma_low=sigma_low, 
                                sigma_high=sigma_high, max_retries=max_retries)
            x_t, fscore_list = parallel_evaluate(x_t)
            max_index, xbest_cost = max(enumerate(fscore_list), key=lambda x: x[1])
            xbest = x_t[max_index]
            evaluations = (idx+1)*population_size
            progress_callback(llm_evo, evaluations)
            
    except KeyboardInterrupt:
        ray.shutdown()

    print("!!! OPTIMIZATION COMPLETE !!!")
    print(f"Best cost: {xbest_cost:.4f}")
    print()

    # pause for a bit to let any CUDA-using processes clean up
    time.sleep(1.0)

    # save the best merge configuration using original model references
    genome_pretty = ModelGenome(config.genome, trust_remote_code=trust_remote_code)
    best_config = genome_pretty.genotype_merge_config(xbest)
    print("Best merge configuration:")
    print(best_config.to_yaml())

    if save_final_model:
        print("Saving final model...")
        run_merge(best_config, os.path.join(storage_path, "final_model"), merge_options)


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    merged = model.merged(
        cache_dir=merge_cache,
        trust_remote_code=trust_remote_code,
    )
    out_path = os.path.join(
        storage_path,
        "input_models",
        merged.model._unique_id(),
    )

    if os.path.exists(out_path):
        logging.info(f"Using existing resharded model at {out_path}")
        return ModelReference(model=out_path)

    model_hf = transformers.AutoModelForCausalLM.from_pretrained(
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join(storage_path, "transformers_cache"),
    )
    model_hf.save_pretrained(
        out_path, safe_serialization=True, out_shard_size=1_000_000_000_000
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.model.path,
            revision=model.model.revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logging.warning(f"Could not save tokenizer for {model.model}", exc_info=e)

    return ModelReference(model=out_path)


if __name__ == "__main__":
    main()

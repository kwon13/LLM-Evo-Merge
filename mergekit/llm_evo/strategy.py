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

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import json
import openai
from openai import OpenAI

import lm_eval.tasks
import numpy as np
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch

from mergekit.llm_evo.actors import InMemoryMergeEvaluator, OnDiskMergeEvaluator
from mergekit.llm_evo.config import EvolMergeConfiguration
from mergekit.llm_evo.genome import ModelGenome
from mergekit.llm_evo.helpers import evaluate_model_ray, merge_model_ray
from mergekit.options import MergeOptions


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
        task_search_path: Union[str, List[str], None] = None,
        model_storage_path: Optional[str] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.batch_size = batch_size
        self.task_manager = lm_eval.tasks.TaskManager(include_path=task_search_path)
        self.model_storage_path = model_storage_path
        if self.model_storage_path:
            os.makedirs(self.model_storage_path, exist_ok=True)

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        pass


class ActorPoolEvaluationStrategy(EvaluationStrategyBase):
    """
    Uses a fixed-size pool of actors to evaluate genotypes in parallel.
    """

    def __init__(
        self,
        *args,
        in_memory: bool = False,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if in_memory:
            self.actor_cls = InMemoryMergeEvaluator
        else:
            self.actor_cls = OnDiskMergeEvaluator

        self.actor_pool = ray.util.ActorPool(
            [
                self.actor_cls.remote(
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    vllm=vllm,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                )
                for _ in range(self.num_gpus)
            ]
        )

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x),
                genotypes,
            )
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]


@ray.remote
class BufferedRayEvaluationStrategyActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        vllm: bool = False,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        model_storage_path: Optional[str] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.vllm = vllm
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.input_queue = []
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.model_storage_path = model_storage_path
        self._shutdown = False

    async def evaluate_genotype(self, genotype: np.ndarray):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result))
        return await future_result

    async def process_queue(self):
        merging: Dict[ray.ObjectRef, asyncio.Future] = {}
        merged: List[Tuple[asyncio.Future, ray.ObjectRef]] = []
        evaluating: Dict[ray.ObjectRef, asyncio.Future] = {}

        logging.info("Starting processing loop")

        try:
            while not self._shutdown:
                while self.input_queue and (len(merging) + len(merged) < self.num_gpus):
                    genotype, future_result = self.input_queue.pop(0)
                    merging[
                        merge_model_ray.remote(
                            genotype,
                            self.genome,
                            self.model_storage_path,
                            self.merge_options,
                        )
                    ] = future_result

                while merged and len(evaluating) < self.num_gpus:
                    future_result, merged_path = merged.pop()
                    evaluating[
                        evaluate_model_ray.remote(
                            merged_path,
                            self.config.tasks,
                            num_fewshot=self.config.num_fewshot,
                            limit=self.config.limit,
                            vllm=self.vllm,
                            batch_size=self.batch_size,
                            task_manager=self.task_manager,
                        )
                    ] = future_result

                ready, _ = ray.wait(
                    list(merging.keys()) + list(evaluating.keys()),
                    num_returns=1,
                    fetch_local=False,
                    timeout=1,
                )
                for r in ready:
                    if r in merging:
                        future_result = merging.pop(r)
                        merged.append((future_result, r))
                    elif r in evaluating:
                        future_result = evaluating.pop(r)
                        future_result.set_result(await r)

                if (
                    not self.input_queue
                    and not merging
                    and not merged
                    and not evaluating
                ):
                    await asyncio.sleep(1)
        except Exception as e:
            logging.error("Error in processing loop", exc_info=e)
            raise

    async def shutdown(self):
        self._shutdown = True


class BufferedRayEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for buffered mode")

        super().__init__(*args, **kwargs)
        self.actor = BufferedRayEvaluationStrategyActor.options(
            max_concurrency=1000
        ).remote(
            self.config,
            self.genome,
            self.merge_options,
            model_storage_path=self.model_storage_path,
            vllm=vllm,
            num_gpus=self.num_gpus,
            task_manager=self.task_manager,
        )
        self.actor.process_queue.remote()

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return ray.get([self.actor.evaluate_genotype.remote(x) for x in genotypes])

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return ray.get(self.actor.evaluate_genotype.remote(genotype))


@ray.remote
def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
):
    pg = ray.util.placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
    strat = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
        placement_group=pg
    )
    merged_path = merge_model_ray.options(scheduling_strategy=strat).remote(
        genotype, genome, model_storage_path, merge_options
    )
    if not merged_path:
        return {"score": None, "results": None}
    res = ray.get(
        evaluate_model_ray.options(scheduling_strategy=strat).remote(
            merged_path,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            vllm=vllm,
            batch_size=batch_size,
            task_manager=task_manager,
        )
    )
    ray.util.remove_placement_group(pg)
    return res


class SerialEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        self.vllm = vllm
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for serial mode")
        super().__init__(*args, **kwargs)

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return ray.get(
            [
                evaluate_genotype_serial.remote(
                    x,
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    vllm=self.vllm,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                )
                for x in genotypes
            ]
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]


# LLMEvolutionStrategy
class LLMEvolutionStrategy:
    def __init__(self, api_key, model, system_prompt_template, user_prompt_template, dimensions, population_size):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions  
        self.population_size = population_size
        self.system_prompt = system_prompt_template.substitute(dimensions=dimensions)
        self.user_prompt_template = user_prompt_template
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.thought = ''
        self.prev_best_cost = -np.inf
        self.prev_best_genome = None
        self.generation = 0
        
    def mutate(self, pre_gen: np.array, fit_scores: list[float], max_retries: int = 3, sigma_low:float=0.05, sigma_high:float=0.2) -> np.array:
        self.generation += 1
        fitness_improved = max(fit_scores) > self.prev_best_cost
        
        # Scale and discretize pre_gen to integer-based format for LLM compatibility
        pre_gen_list=[]
        for idx, (gen, score) in enumerate(zip(pre_gen, fit_scores)):
            if score > self.prev_best_cost:
                self.prev_best_cost = score
                self.prev_best_genome = gen
            gen = (gen * 100).astype(int).reshape(*self.dimensions).tolist()
            pre_gen_list.append({"fitness_score": score, f"genome_{idx+1}": gen})

        user_prompt = self.user_prompt_template.substitute(
            generation=self.generation,
            prev_generation=pre_gen_list,
            prev_best_cost=self.prev_best_cost
        )
        self.messages.append({"role": "user", "content": user_prompt})

        json_output = self._get_response_with_validation(max_retries)
        self.thought = json_output['thought']

        # Convert the mean genome back to [0, 1] range
        mean_genome = np.array(json_output['mean_genome']) / 100

        # Generate the new population by applying Gaussian mutation to the mean genome
        new_generation = []
        # Apply Gaussian noise with varying sigma depending on fitness improvement
        sigma = sigma_low if fitness_improved else sigma_high
        for _ in range(self.population_size):
            mutated_genome = self.gaussian_mutation(mean_genome, sigma)
            new_generation.append(mutated_genome.flatten())
        return np.stack(new_generation, axis=0)

    def gaussian_mutation(self, genome, sigma):
        mutation = np.random.normal(0, sigma, size=genome.shape)
        mutated_genome = genome + mutation
        return np.clip(mutated_genome, 0, 1)  # Keep values within [0, 1]

    def _get_response_with_validation(self, max_retries):
        retry_count = 0
        while retry_count < max_retries:
            try:
                completion = self.get_completion(self.messages)
                self.messages.append(completion)
                json_output = json.loads(completion['content'])
                dimension_check = self.genome_dimension_check(json_output)
                if dimension_check is True:
                    return json_output
                else:
                    error_message = dimension_check
                    print("GENERATION NOT VALID")
                    self.messages.append({
						"role": "user",
						"content": f"Genome dimensions not valid. Error:\n{error_message}\nPlease regenerate the next mean genome."
					})
                    retry_count += 1
            except openai.error.AuthenticationError as e:
                print(f"Authentication error: {e}")
                raise e  
            except (openai.error.Timeout, openai.error.RateLimitError, openai.error.APIError) as e:
                print(f"OpenAI API error: {e}")
                retry_count += 1
                continue

        raise ValueError("Maximum retries reached. Unable to get a valid response.")

    def genome_dimension_check(self, json_output: dict) -> str:
        genome_array = np.array(json_output['mean_genome'])
        if genome_array.shape == self.dimensions:
            return True
        else:
            return (f"The mean_genome's dimensions {genome_array.shape} do not match the specified dimensions {self.dimensions}.")
                                          
    def get_completion(self, messages):
        return self.client.chat.completions.create(
            model=self.model,
            n=1,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        ).choices[0].message.to_dict()

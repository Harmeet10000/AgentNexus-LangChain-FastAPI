# import dspy
# import gepa
# import gepa.optimize_anything as oa
# from gepa.optimize_anything import EngineConfig, GEPAConfig, optimize_anything


# def evaluate(candidate: str) -> float:
#     result = run_my_system(candidate)
#     oa.log(f"Output: {result.output}")  # Actionable Side Information
#     oa.log(f"Error: {result.error}")  # feeds back into reflection
#     return result.score


# result = optimize_anything(
#     seed_candidate="<your initial artifact>",
#     evaluator=evaluate,
#     objective="Describe what you want to optimize for.",
#     config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
# )

# optimizer = dspy.GEPA(
#     metric=your_metric,
#     max_metric_calls=150,
#     reflection_lm="openai/gpt-5",
# )
# optimized_program = optimizer.compile(student=MyProgram(), trainset=trainset, valset=valset)


# trainset, valset, _ = gepa.examples.aime.init_dataset()

# seed_prompt = {
#     "system_prompt": "You are a helpful assistant. Answer the question. "
#     "Put your final answer in the format '### <answer>'"
# }

# result = gepa.optimize(
#     seed_candidate=seed_prompt,
#     trainset=trainset,
#     valset=valset,
#     task_lm="openai/gpt-4.1-mini",
#     max_metric_calls=150,
#     reflection_lm="openai/gpt-5",
# )

# print("Optimized prompt:", result.best_candidate["system_prompt"])

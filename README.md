# Task Suitability Profiles

This repo contains the code for assessing the suitability of AI systems for tasks and roles in the workforce. It provides a pipeline to:

1. Annotate benchmarks using capability-based profiling.  
2. Evaluate the performance of a chosen AI system on these benchmarks.  
3. Model the **capability profiles** of AI systems from their performance data across cognitive domains.  
4. Assess the suitability of AI systems for workplace tasks or roles, by comparing their capability profiles to importance weightings generated from worker surveys.

## Structure

The repository is divided into two main components:

1. **`Benchmarks`**  
   Contains all code required to:
   - Annotate new benchmarks according to the rubric.  
   - Evaluate an LLM of your choice on these benchmarks.

2. **`Suitability`**  
   Contains all code required to:
   - Transform raw benchmark results into capability profiles.  
   - Compare these profiles against importance weightings from survey data.

This README focuses on helping you get started and walks through a complete worked example. Each component also includes its own detailed README describing available configuration options and how to extend the system -- for example, by adding new benchmarks or incorporating additional survey data.

## Getting started

### setup
The first step is to create a virtual environment either using `venv` or `uv`.
```bash
# for venv
python -m venv .venv
source .venv/bin/activate
# for uv 
uv venv
```
Next you will need to install all of the requirements for the repo
```bash
# for venv
pip install -e .
# for uv
uv sync 
```
If you wish to run any AI systems using an API you will need to set the API key for that model, this is provider dependent but generally will look something like.
```bash
export OPENAI_API_KEY=<your api key here>
``` 
alternatively you can create a `.env` file in the root of this repo and set the api keys in there.

Some benchmarks require downloading datasets from Hugging Face or GitHub. See the [Benchmarks README](Benchmarks/README.md#dataset-access) for the full list and instructions.

### annotate benchmarks
Following recent work (see https://arxiv.org/abs/2503.06378), we use a Large Language Model (LLM) annotator to rate benchmark items according to their cognitive demands as specified by rubrics crafted by domain experts.

We have selected relevant benchmarks for the cognitive capabilities we are evaluating, and have completed a full annotation run of these benchmarks using `openai/gpt-4o` (annotations.csv)

To perform your own annotations run:

```bash
python -m Benchmarks.Annotations.run_annotations --model <model name here> 
# or
uv run -m Benchmarks.Annotations.run_annotations --model <model name here> 
```
Further information regarding benchmark annotation is contained in the subfolder readme.

### evaluate models
Once you have created an annotation or chosen to use the provided default annotations you can start evaluating a model you are interested in. You can manually run any benchmark in the project but we have included a script that will run them for you. To evaluate a model on each of the benchmarks run the following command:
```bash
python -m Benchmarks.run_all_tasks --model <model name here>
# or
uv run -m Benchmarks.run_all_tasks --model <model name here>
```
This will produce a single large csv containing the score the model got on each sample in the benchmarks and is saved to the `./logs/` folder by default, we have again included some results we collected from `openai/gpt-4o-mini` if you don't wish to run the evaluation yourself. There are more options available including all inspect ai arguments but more detail can be found in the other readmes.

### generate capability profiles
Now that we have the data on the models performance plus the annotations for what capabilities are required for each sample we can combine them to produce a capability profile for the model. To do this we can run the following command:
```bash
python -m Suitability.scripts.run_inference --mode llm --results <path to results file> --annotations <path to annotations file> --output <path to output folder> --agent-name <the name for this agent>
# or 
uv run -m Suitability.scripts.run_inference --mode llm --results <path to results file> --annotations <path to annotations file> --output <path to output folder> --agent-name <the name for this agent>
```
We have included the results we collected from gpt-4o-mini and the annotations of each benchmark by gpt-4o at `./Suitability/data/raw/gpt-4o-mini_results.csv` and `./Suitability/data/processed/annotations.csv` respectively. However, you can use your own annotations or results collected from the previous sections if you wish. This will produce a folder containing the capability profile for the evaluated model which can be visualised into graphs using the following command:
```bash
python -m Suitability.scripts.visualize_profiles --agents <list of agent names> --idata-base <path to capability profiles> --output <path to save the figures to>
# or
uv run -m Suitability.scripts.visualize_profiles --agents <list of agent names> --idata-base <path to capability profiles> --output <path to save the figures to>
```

Generating the capability profiles can take a long time to run and so we have included the results from gpt-4o-mini in `./Suitability/data/processed/` for you to use without having to run the inference yourself.

### estimate suitability
Now that we have generated the capability profiles we can calculate how suitable a given model would be to a specific task or role. We also need an ability matrix containing the infomation about which capabilites are needed for a given role or task, we have again included a set ability matricies for you to use at `./Suitability/data/processed/ability_matrix_<domain>.csv` which contains the abilities for a variety of positions.
```bash
python -m Suitability.scripts.run_suitability --agents <list of agent names> --idata-base <path to capability profiles> --ability-matrix <Path to ability matrix>
# or
uv run -m Suitability.scripts.run_suitability --agents <list of agent names> --idata-base <path to capability profiles> --ability-matrix <Path to ability matrix> 
# results are saved to Suitability/data/results/ by default
```

If you wish to produce your own ability matrix for a given role you can use the following command:
```bash
python -m Suitability.scripts.build_ability_matrix --companies <path to csv containing company data> --online <path to csv containing data collected online> --output <path to save ability matrix csv to>
# or
uv run -m Suitability.scripts.build_ability_matrix --companies <path to csv containing company data> --online <path to csv containing data collected online> --output <path to save ability matrix csv to>
```
To collect your own task demand data, use the questionnaire template at `./Suitability/data/raw/questionaire_template.csv`. The Qualtrics survey file (`Future_of_Skills_Questionnaire.qsf`) and the accompanying interview script (`Interview script.pdf`) are included in this repository if you wish to run your own data collection.


# Task Suitability Profiles

This repo contains the code for understanding how suitable different large language models are to different tasks and roles in the workforce. It contains a pipeline for annotating benchmarks using capability-based profiling, evaluating the performance of an LLM on these benchmarks, turning these results into capability profiles showing how good the LLM is in different domains and finally compairing the capability profiles to those required for specific tasks and roles, given human results on the same benchmarks.

## Structure

This repo is split roughly in half with `Benchmarks` containing all of the code required to annotate new benchmarks according to the rubric and then evaluate a LLM of your choice on these benchmarks. The other half is contained in `Suitability` which contains all of the code for turning the raw results from the benchmarks into capabilitiy profiles and then compairing them against results from humans. This readme will focus on showing how to get started and a full worked example but note that there are readmes within each section that explain in further detail all of the different options you can use and how to expand on them such as how to add new benchmarks or new human results.

## Getting started

### setup
The first step in setting up this repo is to create a virtual environment either using `venv` or `uv`.
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
If you wish to run any of the language models using an api you will need to set the api key for that model, this is provider dependent but generally will look something like.
```bash
export OPENAI_API_KEY=<your api key here>
``` 
alternatively you can create a `.env` file in the root of this repo and set the api keys in there

### annotate benchmarks
To create a set of annotations for the benchmark we have included a script that will evaluate each item in the benchmarks according to the rubrics specified in the rubrics folder. To create the annotations run the following command:

```bash
python -m Benchmarks.Annotations.run_annotations --model <model name here> 
# or
uv run -m Benchmarks.Annotations.run_annotations --model <model name here> 
```
This can take a long time to run and cost alot of tokens as each item in the benchmarks must be processed one per dimension in the rubric. To help with this we have included the results from an annotation run by `openai/gpt-4o` as the annotations csv file in the root of each task directory, new annotations will be created in a subfolder of each task directory.

There are many more options to control the annotation process but they have been set to reasonable defaults, if you want more information look at the other readmes in the subfolders of this repo.


### evaluate models
Once you have created an annotation or chosen to use the provided default annotations you can start evaluating a model you are interested in. You can manually run any benchmark in the project but we have included a script that will run them for you. To evaluate a model on each of the benchmarks run the following command:
```bash
python -m Benchmarks.run_all_tasks --model <model name here>
# or
uv run -m Benchmarks.run_all_tasks --model <model name here>
```
This will produce a single large csv containing the score the model got on each sample in the benchmarks and is saved to the `./logs/` folder by default, we have again included some results we collected from `openai/gpt-4o-mini` if you don't wish to run the evaluation yourself. There are more options available including all inspect ai arguments but more detail can be found in the other readmes.

### calculate capability profiles
Now that we have the data on the models performance plus the annotations for what capabilities are required for each sample we can combine them to produce a capability profile for the model. To do this we can run the following command:
```bash
python -m Suitability.scripts.run_inference --mode llm --results <path to results file> --annotations <path to annotations file> --output <path to output folder> --agent-name <the name for this agent>
# or 
uv -m Suitability.scripts.run_inference --mode llm --results <path to results file> --annotations <path to annotations file> --output <path to output folder> --agent-name <the name for this agent>
```
We have included the results we collected from gpt-4o-mini and the annoations of each benchmark by gpt-4o at `./Suitability/data/raw/gpt-4o-mini_results.csv` and `./Suitability/data/processed/annotations.csv` respectively. However, you can use your own annotations or resutls collected from the previous sections if you wish. This will produce a folder containing the capabilitiy profile for the evaluated model which can be visualised into graphs using the following command:
```bash
python -m Suitability.scripts.visualize_profiles --agents <list of agent names> --idata <path to capability profiles> --output <path to save the figures to>
# or
uv run -m Suitability.scripts.visualize_profiles --agents <list of agent names> --idata <path to capability profiles> --output <path to save the figures to>
```

Generating the capability profiles can take a long time to run and so we have included the results from gpt-4o-mini in the folder `./Suitability/data/results` for you to use without having to run the inference yourself.

### calculate suitability
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
We collcted data from both individual companies as well as online questionaires. we cannot include that data here but if you wish to collect your own data and use it to build an ability matrix please use the format used in the template at `./Suitability/data/raw/questionaire_template.csv`


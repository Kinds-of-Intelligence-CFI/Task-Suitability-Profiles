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

### calculate suitability
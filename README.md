# Task Suitability Profiles

This repo contains the code for understanding how suitable different large language models are to different tasks and roles in the workforce. It contains a pipeline for annotating benchmarks using capability-based profiling, evaluating the performance of an LLM on these benchmarks, turning these results into capability profiles showing how good the LLM is in different domains and finally compairing the capability profiles to those required for specific tasks and roles, given human results on the same benchmarks.

## Structure

This repo is split roughly in half with `Benchmarks` containing all of the code required to annotate new benchmarks according to the rubric and then evaluate a LLM of your choice on these benchmarks. The other half is contained in `Suitability` which contains all of the code for turning the raw results from the benchmarks into capabilitiy profiles and then compairing them against results from humans. This readme will focus on showing how to get started and a full worked example but note that there are readmes within each section that explain in further detail all of the different options you can use and how to expand on them such as how to add new benchmarks or new human results.

## Getting started

### setup
The first step in setting up this repo is to create a virtual environment either using `venv` or `uv`.
```bash
# for venv
python -m venv venv
# for uv 
uv init
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

### evaluate models

### calculate capability profiles

### calculate suitability
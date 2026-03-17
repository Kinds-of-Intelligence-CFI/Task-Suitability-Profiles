# **Planning**

This rubric assesses how much a task requires the respondent to formulate and lay out a strategy or a sequence of steps, deciding what actions to take, in what sequence, sometimes with contingencies, to reach a stated goal. Difficulty scales from one‑shot answers (no real plan) to long‑horizon, multi‑layer strategies that juggle uncertainty, optimisation, and interdependent sub‑goals.

## **Level 0: None**

The task’s “time horizon” is a single move: recall a fact, translate a sentence, or make one selection. No decomposition, ordering, or resource scheduling enters the picture.

**Examples:**

* In what year did Darwin publish *On the Origin of Species*? (1856 / 1859 / 1865 / 1866 )  
* “Define the term *algorithm*.”  
* “Name the capital of Spain.”

## **Level 1: Very Low**

A minimal, obvious two‑ or three‑step plan is required; nearly anyone can state the steps in correct order without considering constraints.

**Examples:**

* Describe, in order, how to send an email with an attachment.  
* Write a three‑step plan to copy a file, rename it, and email it to yourself.  
* Write a one‑sentence plan for boiling pasta.

## **Level 2: Low**

Either (a) \< 3 subtasks that can each be solved by a Level‑1 plan, or (b) a linear sequence of \< 10 well‑defined actions executed under one or two explicit rules. Environment behaviour is fully known and deterministic.

**Examples:**

* Draft a step‑by‑step agenda for a 60‑minute team meeting that includes: (a) three discussion topics, (b) a timed breakout session, (c) a ten‑minute Q\&A at the end, ensuring the schedule fits exactly in the hour.  
* Provide a sequence of actions to create a webpage at [https://kinds-of-intelligence-cfi.github.io/ADELE/](https://kinds-of-intelligence-cfi.github.io/ADELE/)  
* Provide an ordered list of shell commands to create a directory, move two files into it, and compress the directory into a zip archive.

## **Level 3: Intermediate**

Plans span \> 20 actions and must accommodate unknowns such as resource availability or stochastic outcomes. The high-level goal is clear, but multiple viable routes are possible.

**Examples:**

* Design a solution to transfer four disks in the Tower of Hanoi puzzle from peg A to peg C using standard rules.  
* Draft a step‑by‑step agenda for a 60‑minute team meeting that includes: (a) three discussion topics, (b) a timed breakout session, (c) a ten‑minute Q\&A at the end, ensuring the schedule fits exactly in the hour.  
* Plan a 3-day mini data-science project to deliver a churn-prediction baseline under time and resource constraints. A potential solution might involve pulling and joining customer, usage, and billing tables; defining the churn target and splitting the data; running exploratory analysis; engineering features; training simple models; and producing both performance metrics and a short stakeholder readout. 

## **Level 4: High**

Tasks at this level require detailed planning for more than 20 steps within tasks or subtasks, nested across up to three levels of goals and subgoals. The overall goal may be open-ended, and while the environment rules may not be fully known in advance, they remain static once discovered. Effective planning often involves coordinating different levels of abstraction: setting broad strategies at the top level while elaborating detailed sequences for near-term or complex subtasks.

**Examples:**

* Draft the full workflow to produce a high-quality AI research paper on the topic of AI evaluation. It should include literature survey, research questions, dataset curation, experimental design, statistical analysis, writing and submission. Include all tasks and subtasks, dependencies between then and and a timeline.  
* Outline a CI/CD pipeline for a multi-repo research stack (data-prep, model-training, and API layers): build the Python data-prep package and Rust inference microservice in parallel on a two-core runner, cache Docker layers, trigger GPU model-training only after both builds and their unit tests pass, generate Sphinx docs and experiment reports, run integration tests against a transient Postgres service, then push signed images to staging and, after automated latency benchmarks beat the previous release, promote to production with a blue-green rollout—all steps scheduled to minimise total wall-clock time.  
* Develop a five-day international AI evaluation workshop hosted in Valencia for 200 attendees. Secure an accessible venue with three parallel rooms, draft a schedule that balances two daily keynotes, four parallel technical tracks, poster sessions, and a nightly networking event; coordinate catering that covers vegan, halal, and gluten-free options; arrange simultaneous-interpretation services in Spanish and English; build a registration timeline with early-bird, regular, and on-site tiers; budget for travel grants, equipment rental, and marketing; plot ground-transport links from airport to hotels; and include contingency buffers for speaker cancellations and over-capacity rooms.

## **Level 5: Very High**

Plans contain \> 3 nested layers of tasks/subtasks over months‑to‑years horizons, fast moving targets and sparse feedback. The environment is dynamic, feedback is sparse, and success hinges on adaptive replanning, resource allocation, and balancing competing objectives while maintaining coherence across the entire structure.

**Examples:**

* Work out a viable plan to take a ten‑person startup from zero revenue to $10 million ARR in three years: craft product roadmap, fund‑raising strategy, hiring plan, go‑to‑market sequencing, and competitor‑response playbooks.  
* Design the full narrative bible, world‑building atlas, and character arcs for a Tolkien‑scale epic trilogy, including linguistic systems, myth cycles, political history, and inter‑book foreshadowing cues.  
* Design a 10-year, multinational program that captures a 50-ton near-Earth asteroid, transports it to cis-lunar space, and establishes a commercially viable resource-extraction outpost.
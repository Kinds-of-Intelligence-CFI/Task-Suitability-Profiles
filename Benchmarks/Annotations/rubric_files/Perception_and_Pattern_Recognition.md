# **Perception and Pattern Recognition**

This rubric assesses how much a task depends on detecting and interpreting salient details, e.g. signals, regularities, anomalies, or subtle cues, by drawing on prior knowledge. At the easiest levels, surface patterns are obvious and almost everyone spots them. At the hardest, success demands synthesising faint, multi-layered regularities buried in noisy textual data or narratives that only a tiny fraction of people can reliably discern.

## **Level 0: None**

The task needs no pattern recognition; copying or basic recall suffices.  Performance is unaffected by ignoring all contextual or relational cues.

**Examples**:

* Copy a provided numerical string exactly.  
* Convert “hello” to uppercase.  
* Select answer ‘C’ because the prompt explicitly says “Choose C”.

## **Level 1: Very Low**

Requires noticing a single, highly salient cue or simple pattern that is obvious to most people. Recognition is nearly automatic for most readers.

**Examples**:

* In “The meeting is on Friday at 10 am,” extract the day of the week  
* Identify which of the following three lines of code contains the repeated trigram ‘abc’.  
* Circle the red triangle in a picture containing three blue circles and one red triangle.

## **Level 2: Low**

The solver must recognise a simple but non-obvious pattern, for instance, a basic trend or a two-feature correlation, across a short span of data. A small amount of prior knowledge or inference improves performance, but the rule remains concrete and explicit once seen.

**Examples**:

* From a six-row sales table, identify the first month where revenue begins a consistent decline.  
* In a page of 20 digits, locate the one instance of “2” among otherwise similar-shaped “5”s.  
* Given ten timestamped log entries, mark the three that break an every-15-minutes rhythm.

## **Level 3: Intermediate**

Tasks require integrating multiple cues or cross-referenced details, often with distractors, to spot an anomaly, hidden rule, or foreshadowing device. The pattern is discoverable but rarely obvious.

**Examples:**

* Examine ten HTTP logs and identify the single entry that signals a breach by combining an unusual status code with an out-of-range timestamp.  
* In a 500-word short story, highlight the sentence that foreshadows the twist ending via a double meaning.  
* From a lineup of six nearly identical suitcases moving on a baggage carousel, identify the correct one by piecing together two or more subtle features (e.g., faded logo plus mismatched zipper pull), while ignoring false matches that share only one of those features.

## **Level 4: High**

The solver must uncover layered or shifting patterns within noisy, partially relevant data, often merging domain knowledge with logical elimination. Cues may conflict or be sparsely distributed, and several plausible distractions exist.

**Examples:**

* Analyse a 40-line Slack thread to deduce which employee leaked data, using subtle time-zone hints and vocabulary idiosyncrasies.  
* From a summary of 30 experiments, identify a hidden confounder that causes all trials with a particular protocol to fail only under humidity \> 80%.  
* In an aerial photo of a dense city block, identify the one building that violates planning regulations by comparing shadow lengths and roof angles against orientation markers.

## **Level 5: Very High**

Success demands detecting faint, multi-layer regularities or covert structures across lengthy, noisy data where patterns nest, shift or are adversarially obfuscated.

**Examples**:

* From a 10000-word email archive, trace an insider-trading scheme by linking coded phrases, irregular send times, and repeating checksum strings.  
* Analyse a 50-page detective story told in shuffled diary snippets to identify the true culprit, using mirrored dates and inconsistent pronouns.  
* From a 1,000-frame satellite image sequence, detect a covert military build-up by linking faintly altered shadow orientations, irregular vehicle spacing, and camouflage patterns that change between frames.


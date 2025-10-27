# FiGO
Code for paper - "FiGO: Fine-Grained Query Optimization for Video Analytics" (SIGMOD'2022).

## Installation 
```bash
conda env create -f environment.yml
```

## Hardware
* TITAN Xp + CUDA 11.6
Ideas can be tested as well by using cached inference results.

## Example Run
The easiest way to evaluate the idea of FiGO with other systems is to use cached results (avoid extra inference time).
```bash
# run q1 end-to-end execution with all system implementations
python experiment/end_to_end.py --cache-path $PWD/demo-res --query q1
```
Accuracy (F-1 score) and query processing time should be avaiable at `res/q1/res.txt`.

## Currently Supported Systems
* FiGO
* MS-Filter: Filter-based approach
* ME-Coarse: Coarse-grained optimization that entire video uses a single model
* MC: Model cascade approach
* Naive: Run the most accuracy model on all frames. Used for accuracy evaluation.

## Documentation
* Every system implements a scheduler, which does both query optimization and execution. The scheduler will take a loader to load decoded frames of a video.
* Query directory contains queries implemented by using provided scheduler and predicates. 
* Experiment folder contains an end-to-end script to run all systems and collect results. 

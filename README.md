# Cell-Tracker

## Problem Definition
* There are three datasets which contains time-lapse image frame of biological cells in datasets. They are named DIC-C2DH-HeLa, Fluo-N2DL-HeLa, and PhC-C2DL-PSC. 
* Implement an automated detecting and tracking tool for these three datasets. It should draw a bounding box around each cell, draw the trajectory for each cell, and be capable to detect cell division events.
* Calculate speed of the cell, total distance travelled, net distance travelled, and confinement ratio for each cell at each frame for analysis purposes.

## Evaluation and Results
* Detection Module: Average accuracy -- 92%
* Trajectory: Average accuracy -- 88%
* Mitosis detection: Average accuracy -- 86%

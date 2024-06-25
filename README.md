# Video Searcher - Searching and Indexing Video with Input Clip
A video query system capable of identifying the source video and timestamp of input videos from a database using content-based retrieval, even if input videos are filtered or differ in resolution, with a sub-2-second response time. Implemented MFCC audio feature extraction and image hashing to create an accurate video identification algorithm. 

The project is designed to take an input video clip along with its audio and identify which video from a database it originates from. Additionally, the program will determine the exact starting point of the clip within the original video.

## Setup
Please ensure videos are stored under data/videos and that conda is already installed! In the root directory of this repository, run the following:

```bash
conda env create -f environment.yml
conda activate multimedia
```

## Usage
main.py : Main file used to run program. Prompts user for input query video filepath via terminal, prints out results, and displays original video. Once the terminal prompts for a query video, simply input the path to the query video and it should output the result on your browser.<br />

```bash
python3.10 main.py
```
<br />
data/ : Contains input query videos, original videos, and precomputed data.<br />
vindex.py/ : Contains helper functions and class managing algorithm<br />
templates/ : Contains html for web player

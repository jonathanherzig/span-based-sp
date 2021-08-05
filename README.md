# Span-based Semantic Parsing for Compositional Generalization

Author implementation of the following [ACL 2021 paper](https://aclanthology.org/2021.acl-long.74.pdf).

## Setup

1. Install an Anaconda virtual environment:
	```
	conda create -n compgen python=3.7 anaconda
    ```
2. Install the following pip version
    ```
	  pip install pip==20.0.2
    ``` 
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Install swi-prolog with version <8 (this is a Prolog interpreter used to execute GeoQuery). For example, for Ubuntu follow [these instructions]([https://www.howtoinstall.me/ubuntu/18-04/swi-prolog/](https://www.howtoinstall.me/ubuntu/18-04/swi-prolog/)).
5. Download the [datasets](https://drive.google.com/file/d/1Srs_7hKtKhTunNWg1ThOlVo-rFFwFEq6/view?usp=sharing) and extract the content to `span-based-sp/datasets` 

## Training and evaluating the span-based parser
The main script for running experiments is `span-based-sp/run_span_exps.sh`. To run an experiment use:
```
bash run_span_exps.sh domain
```
Where `domain=[geo, scan, clevr]`. A single experiment will train over all splits experimented in the paper for the specific domain. 
We use early stopping w.r.t denotation accuracy on the dev set. Results are written to a log file under the `logs/` folder.

To conveniently print all logs, run:
```
python utils/combine_logs.py
```

## Environment & Setup
Engine: Elasticsearch 9.3.2 (Docker).

Python Client: elasticsearch

## Dataset 
- Using a small dataset to set (87MB)
- https://huggingface.co/datasets/rahular/simple-wikipedia
- saved it as a lcocal parquet file
- it doesnt have titles so its a problem (maybe?)
- creating a fake title out of the first line

## Index
- use a part of the dataset for now (the whole one takes a couple minutes)
- use a generator to not store the entire dataset in RAM (for the future with a bigger dataset)
- use bulk API to index a batch and not index row by row 
- yield -> helper.bulk collects 500 documents -> sends a request to elastic search to index

## Results 
- queried the word april
- output: Title: On 4 April these restrictions were extended until 27 April a... (Score: 6.714231)
Title: He was elected in for the term April 2014-April-2020.... (Score: 6.629226)
Title: Neillsville was platted on April 14, 1855 and incorporated i... (Score: 6.557351)
Title: April... (Score: 6.5244484)
Title: April... (Score: 6.5244484)

## Run
- start elastic search container
- start kibana container
- run ```python app.py``` for the UI
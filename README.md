# Sentence Transformer

## Run offline ingestion 

To run the application for offline ingestion use the following command:

```bash
sh build_offline_run.sh <path_to_folder>  
````

> <path_to_folder> folder path where to mount the volume of application from docker image. This will save the results as output.json file in the folder mounted.
---


## Run Online ingestion (This can be ignored just did for fun)

To run the application for online ingestion from api use the following command:

```bash
sh build_online_run.sh
````

It will run the server at `http://127.0.0.1:5007` and to get the result just run `http://127.0.0.1:5007/api/007`. Change the 007 to any doc_id as desired.

## `src/offline_ingestion.py`

This file contains the function to do the batch ingestion mechanism for text encoders.
This file will save the results in output/output.json file

> By default I am running on small subset of `doc_id=007` you can change to `12345` if needed.

## `src/util.py`

This file contains all the functions needed for offline ingestion and online ingestion for text encoders. 
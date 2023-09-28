# 9fin

## Run

To run the application use the following command:

```bash
sh build_run.sh <path_to_folder>  
````

> <path_to_folder> folder path where to mount the volume of application from docker image.
---

## `src/offline_ingestion.py`

This file contains the function to do the batch ingestion mechanism for text encoders.
This file will save the results in output/output.json file

> By default I am running on small subset of `doc_id=007` you can change to `12345` if needed.

## `src/util.py`

This file contains all the functions needed for offline ingestion and online ingestion for text encoders. 
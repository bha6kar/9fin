volume_path=$1
docker build -t offline-ingestion .
docker run -it --rm --volume $volume_path:/app/output offline-ingestion

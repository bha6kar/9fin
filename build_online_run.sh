docker build -t online_ingestion_api -f Dockerfile_online .
docker run -p 5007:5007 online_ingestion_api

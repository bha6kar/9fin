# Use a base image (e.g., Python)
FROM python:3.8-slim

COPY app/entrypoint1.py /app/
COPY app/entrypoint2.py /app/

COPY app/entrypoint21.py /app/

# Set the first entry point
# ENTRYPOINT [, "/app/entrypoint2.py"]

# Set the second entry point
CMD ["/app/entrypoint21.py", "python"]

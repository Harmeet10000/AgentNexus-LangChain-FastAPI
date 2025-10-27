

# Build the Docker image (using the project's Dockerfile)
docker build -t langchain-fastapi-server -f docker/dev.Dockerfile .

# Run the Docker container (detached, port 5000)
docker run -d \
  -p 5000:5000 \
  --name langchain-fastapi-server \
  --env-file .env.development \
  langchain-fastapi-server:latest

# View logs
docker logs -f langchain-fastapi-server

# Tag for Docker Hub
docker tag langchain-FastAPI-server harmeet10000/langchain-fastapi-server:latest

# Push to Docker Hub
docker push harmeet10000/langchain-fastapi-server:latest

# (For AWS ECR push replace the tag above with your ECR repo and push)
docker push <your-ecr-repo-uri>

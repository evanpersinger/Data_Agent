# Useful Docker Commands

## Starting and Stopping Services

```bash
# Start all services in detached mode
docker-compose up -d

# Start all services and view logs
docker-compose up

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v

# Restart a specific service
docker-compose restart data-agent
docker-compose restart postgres
```

## Viewing Logs

```bash
# View logs for all services
docker-compose logs

# View logs for a specific service
docker-compose logs data-agent
docker-compose logs postgres

# Follow logs in real-time
docker-compose logs -f data-agent

# View last 100 lines of logs
docker-compose logs --tail=100 data-agent
```

## Accessing Containers

```bash
# Access data-agent container shell
docker exec -it data-agent /bin/bash

# Access postgres container shell
docker exec -it data-agent-postgres /bin/bash

# Run a command in data-agent container without entering shell
docker exec data-agent python data_agent.py

# Run a Python command in data-agent container
docker exec -it data-agent python -c "print('Hello')"
```

## Database Access

```bash
# Connect to PostgreSQL from host machine
# (PostgreSQL is exposed on port 5433)
psql -h localhost -p 5433 -U postgres -d data_agent

# Connect to PostgreSQL from inside data-agent container
docker exec -it data-agent-postgres psql -U postgres -d data_agent

# Run SQL query directly
docker exec -it data-agent-postgres psql -U postgres -d data_agent -c "SELECT * FROM your_table;"
```

## Rebuilding Containers

```bash
# Rebuild containers without cache
docker-compose build --no-cache

# Rebuild and restart services
docker-compose up --build

# Rebuild only data-agent service
docker-compose build data-agent
```

## Container Management

```bash
# View running containers
docker-compose ps

# View all containers (including stopped)
docker ps -a

# View container resource usage
docker stats

# Stop a specific container
docker-compose stop data-agent

# Start a specific container
docker-compose start data-agent

# Remove a specific container
docker-compose rm data-agent
```

## Volume Management

```bash
# List volumes
docker volume ls

# Inspect a volume
docker volume inspect data_agent_postgres_data

# Remove unused volumes
docker volume prune
```

## Network Management

```bash
# List networks
docker network ls

# Inspect network
docker network inspect data-agent_data-agent-network
```

## Useful Debugging Commands

```bash
# Check if containers are healthy
docker-compose ps

# View container environment variables
docker exec data-agent env

# View container configuration
docker inspect data-agent

# Execute Python with debugging
docker exec -it data-agent python -i
```

## Quick Reference

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Access data-agent shell
docker exec -it data-agent /bin/bash

# Stop everything
docker-compose down
```

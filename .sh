#!/bin/bash

# Check if name_project argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <name_project>"
    exit 1
fi

# Assign input argument to variable
name_proj=$1

# Convert project name to lowercase for the Docker registry
name_proj_lc=$(echo "$name_proj" | tr '[:upper:]' '[:lower:]')

# Base directory
BASE_DIR="${name_proj}/src"

# Print the base directory
echo "Base directory is: ${BASE_DIR}"

# Docker registry
DOCKER_REGISTRY="registry.i-soft.com.vn"



# Iterate through each directory in the base directory
for folder in ${BASE_DIR}/*/; do
    # Change to the base directory
    cd "${name_proj}"
    # Extract folder name
    folder_name=$(basename "$folder")
    # Convert folder name to lowercase
    folder_name_lc=$(echo "$folder_name" | tr '[:upper:]' '[:lower:]')

    # Print the full path of the folder being processed
    echo "Processing folder: ${BASE_DIR}/${folder_name}"
    
    echo "Building Docker image for folder ${folder_name_lc}..."
    # Build Docker image using the full path to the Dockerfile
    docker build -t ${folder_name_lc}:staging -f "src/${folder_name}/Dockerfile" .
    
    echo "Tagging Docker image ${folder_name_lc}:staging as ${DOCKER_REGISTRY}/${name_proj_lc}${folder_name_lc}:staging..."
    # Tag the Docker image
    docker tag ${folder_name_lc}:staging ${DOCKER_REGISTRY}/${name_proj_lc}${folder_name_lc}:staging
    
    echo "Pushing Docker image ${DOCKER_REGISTRY}/${name_proj_lc}${folder_name_lc}:staging to the registry..."
    # Push the Docker image to the registry
    docker push ${DOCKER_REGISTRY}/${name_proj_lc}${folder_name_lc}:staging
    
    echo "Completed operations for folder ${folder_name_lc}."
done

#!/bin/bash

# Create workspace directory if it doesn't exist
mkdir -p /workspace
cd /workspace

# Function to download from a Zenodo record
download_zenodo() {
    local record="$1"
    local record_dir="/workspace/$record"
    
    echo "Processing Zenodo record: $record"
    
    # Create directory for this record
    mkdir -p "$record_dir"
    
    # Get the record metadata
    echo "Fetching metadata for record $record..."
    response=$(curl -s "https://zenodo.org/api/records/$record")
    
    # Check if response is valid JSON
    if ! echo "$response" | python3 -c "import json,sys; json.load(sys.stdin)" > /dev/null 2>&1; then
        echo "Error: Invalid response for record $record"
        return 1
    fi
    
    # Extract and download each file
    echo "$response" | python3 -c '
import json
import sys
import os

try:
    data = json.load(sys.stdin)
    files = data.get("files", [])
    
    if not files:
        print("No files found in record", file=sys.stderr)
        sys.exit(1)
        
    # Print record title for folder naming
    metadata = data.get("metadata", {})
    title = metadata.get("title", "Unknown")
    print("Title:", title, file=sys.stderr)
    
    for file in files:
        url = file["links"]["self"]
        name = file["key"]
        size = file.get("size", "Unknown")
        print(f"{url}\t{name}\t{size}")
except Exception as e:
    print("Error processing JSON:", str(e), file=sys.stderr)
    sys.exit(1)
' | while IFS=$'\t' read -r url name size; do
        echo "Downloading $name (Size: $size bytes)"
        
        # Create subdirectories if filename contains paths
        file_dir=$(dirname "$record_dir/$name")
        mkdir -p "$file_dir"
        
        # Download the file
        curl -L --fail --create-dirs -o "$record_dir/$name" "$url"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully downloaded: $name"
        else
            echo "✗ Failed to download: $name"
        fi
    done
    
    echo "Completed processing record: $record"
    echo "Files downloaded to: $record_dir"
    echo "----------------------------------------"
}

# List of records to download
records=(
    "6408497"
    "10656052"
    "8278563"
)

# Process each record
for record in "${records[@]}"; do
    download_zenodo "$record"
done

echo "All downloads completed!"
echo "Check /workspace for downloaded files"
ls -R /workspace
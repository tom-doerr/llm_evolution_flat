#!/bin/bash

while true
do
    FILE="main.py"  # Replace with your file name
    date
    LINES=$(wc -l < "$FILE")
    echo "$FILE has $LINES lines"
    MIN_LINES=100

    # Check line count
    if [ "$LINES" -lt "$MIN_LINES" ]; then
        echo "ERROR: $FILE has fewer than $MIN_LINES lines!"
        echo "Showing context around the file:"
        ls -la $(dirname "$FILE")
        echo "First 10 lines of $FILE:"
        head -10 "$FILE"
        echo "Last 10 lines of $FILE:"
        tail -10 "$FILE"
        echo "Reverting to last commit..."
        git checkout HEAD -- "$FILE"
        echo "File restored from git."
    fi
    
    sleep 1
done

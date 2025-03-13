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
        echo "Reverting to last commit..."
        git checkout HEAD -- "$FILE"
        echo "File restored from git."
    fi
    
    # Run pylint and syntax check
    echo "Running code quality checks..."
    pylint --msg-template '{path}:{line}:{column}: {msg} ({symbol}) [{obj}]' "$FILE" | tail -n 400 > issues.txt.tmp
    python3 -m py_compile "$FILE" 2>&1 | tail -n 400 >> issues.txt.tmp
    mv issues.txt.tmp issues.txt
    echo "Code quality check results saved to issues.txt"
    
    sleep 1
done

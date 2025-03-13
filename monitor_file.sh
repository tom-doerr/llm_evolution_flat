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
    
    # Run syntax check and capture any errors
    SYNTAX_CHECK=$(python3 -m py_compile "$FILE" 2>&1)
    if [ $? -ne 0 ]; then
        echo "$SYNTAX_CHECK" >> issues.txt.tmp
        
        # Extract line number from syntax error message
        if [[ "$SYNTAX_CHECK" =~ line\ ([0-9]+) ]]; then
            ERROR_LINE=${BASH_REMATCH[1]}
            echo -e "\nCode context around syntax error (line $ERROR_LINE):" >> issues.txt.tmp
            
            # Calculate context range (5 lines before and after)
            START=$((ERROR_LINE - 5))
            END=$((ERROR_LINE + 5))
            
            # Ensure start line is at least 1
            if [ $START -lt 1 ]; then
                START=1
            fi
            
            # Add line numbers and extract the code context
            sed -n "${START},${END}p" "$FILE" | nl -v "$START" -w 4 -s ': ' >> issues.txt.tmp
        fi
    else
        echo "No syntax errors detected." >> issues.txt.tmp
    fi
    
    mv issues.txt.tmp issues.txt
    echo "Code quality check results saved to issues.txt"
    
    sleep 1
done

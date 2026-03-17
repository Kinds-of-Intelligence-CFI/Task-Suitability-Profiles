#!/usr/bin/env python3
"""
Convert markdown rubric files to JSON format.
Creates a JSON file with an array where each item has:
- dimension: the title/dimension name (first heading in the markdown)
- rubric: the full markdown content
"""

import os
import json
import re
from pathlib import Path
import argparse


def extract_dimension_name(markdown_content):
    """Extract the first heading from markdown content as the dimension name."""
    lines = markdown_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # Remove markdown heading syntax and clean up
            dimension = re.sub(r'^#+\s*', '', line)
            # Remove any bold formatting
            dimension = re.sub(r'\*\*(.*?)\*\*', r'\1', dimension)
            return dimension.strip()
    
    # Fallback: use filename without extension
    return "Unknown Dimension"


def convert_rubrics_to_json(input_dir, output_file="rubric.json"):
    """
    Convert all .md files in input_dir to a JSON array format.
    
    Args:
        input_dir: Directory containing .md files
        output_file: Output JSON filename
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_dir} does not exist")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a directory")
    
    rubrics = []
    
    # Find all .md files in the directory
    md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return
    
    print(f"Found {len(md_files)} markdown files:")
    
    for md_file in md_files:
        print(f"  Processing: {md_file.name}")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            dimension = extract_dimension_name(content)
            
            rubric_entry = {
                "dimension": dimension,
                "rubric": content
            }
            
            rubrics.append(rubric_entry)
            print(f"    Dimension: {dimension}")
            
        except Exception as e:
            print(f"    Error processing {md_file.name}: {e}")
            continue
    
    # Sort by dimension name for consistency
    rubrics.sort(key=lambda x: x['dimension'])
    
    # Write JSON output - relative to current working directory
    output_path = Path(output_file)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rubrics, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully created {output_path}")
        print(f"Converted {len(rubrics)} rubric files to JSON format")
        
    except Exception as e:
        print(f"Error writing JSON file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert markdown rubric files to JSON format')
    parser.add_argument('--input_dir', help='Directory containing .md files')
    parser.add_argument('-o', '--output', default='rubric.json', 
                        help='Output JSON filename (default: rubric.json)')
    
    args = parser.parse_args()
    
    try:
        convert_rubrics_to_json(args.input_dir, args.output)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
import re
import sys
import os

def remove_comments(file_path, output_path=None):
    """
    Remove all comments from a Python file while preserving code functionality.
    
    Args:
        file_path: Path to the Python file to process
        output_path: Path to save the cleaned file (if None, will use original name with _clean suffix)
    
    Returns:
        Path to the cleaned file
    """
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_clean{ext}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First, handle docstrings and multi-line string literals
    # We need to be careful not to remove string literals that are part of the code
    def replace_triple_quotes(match):
        s = match.group(0)
        # If it's an assignment or part of a function call, keep it
        if re.search(r'=\s*[rfub]*("""|\'\'\')', s) or re.search(r'\([rfub]*("""|\'\'\')', s):
            return s
        # Otherwise, it's likely a comment/docstring, so remove it
        return ''
    
    # Handle triple-quoted strings (both """ and ''')
    pattern = r'("""[\s\S]*?""")|(\'\'\'[\s\S]*?\'\'\')'
    content = re.sub(pattern, replace_triple_quotes, content)
    
    # Handle single-line comments, but be careful not to remove # inside strings
    lines = content.split('\n')
    cleaned_lines = []
    
    in_string = False
    string_char = None
    
    for line in lines:
        cleaned_line = ""
        i = 0
        while i < len(line):
            # Check for string start/end
            if line[i] in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = line[i]
                elif line[i] == string_char:
                    in_string = False
                cleaned_line += line[i]
            # Check for comments outside of strings
            elif line[i] == '#' and not in_string:
                break  # Ignore rest of the line
            else:
                cleaned_line += line[i]
            i += 1
        
        # Only add non-empty lines to preserve code structure
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)
    
    # Write cleaned content to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python comment_remover.py <input_file.py> [output_file.py]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        cleaned_file = remove_comments(input_file, output_file)
        print(f"Comments removed successfully. Cleaned file saved to: {cleaned_file}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()

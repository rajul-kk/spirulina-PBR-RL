import re

def get_body_word_count(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by lines
    lines = content.split('\n')
    
    # Find indices
    start_idx = -1
    refs_idx = -1
    appendix_idx = -1
    
    for i, line in enumerate(lines):
        if "## 1. Environment Overview" in line:
            start_idx = i
        if "## References" in line:
            refs_idx = i
        if "## Appendix A" in line:
            appendix_idx = i
            
    # The body ends at the first of References or Appendix
    end_idx = len(lines)
    if refs_idx != -1 and appendix_idx != -1:
        end_idx = min(refs_idx, appendix_idx)
    elif refs_idx != -1:
        end_idx = refs_idx
    elif appendix_idx != -1:
        end_idx = appendix_idx
        
    if start_idx == -1:
        return 0
        
    # Extract body lines
    body_lines = lines[start_idx:end_idx]
    
    # Count words in body lines
    total_words = 0
    for line in body_lines:
        # We use a simple whitespace split to match standard word counters
        words = line.split()
        total_words += len(words)
        
    return total_words

print(f"Genetic Body Word Count: {get_body_word_count('genetic_env.md')}")
print(f"Light Body Word Count: {get_body_word_count('light_env.md')}")

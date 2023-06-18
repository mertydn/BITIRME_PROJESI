input_file = '/home/mert/pyskl/box.txt'
output_file = '/home/mert/pyskl/box2.txt'
import re

# Read the content of the input file
with open(input_file, 'r') as f:
    content = f.read()
    modified_content = content.replace(']', '],')
    modified_content = re.sub(r'\s+', ' ', modified_content)
    modified_content = re.sub(r'\s+', ',', modified_content)
    modified_content = modified_content.replace(' ', ',')
    modified_content = modified_content.replace(',,', ',')
    modified_content = modified_content.replace('[,', '[')
    modified_content = modified_content.replace('],]', ']]')
    modified_content = modified_content[:-1]

with open(output_file, 'w') as f:
    f.write(modified_content)

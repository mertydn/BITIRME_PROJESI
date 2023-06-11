stgcn_labels_dict = {}
j = 1
with open("/home/oakwood/uniformerv2_demo/nturgbd_120.txt") as f:
    for line in f:
       stgcn_labels_dict[j] = line.rstrip('\n')
       j += 1
       

for i in stgcn_labels_dict:
    print(i, stgcn_labels_dict[i])



import re

output_names2list = []

file_path = "/home/oakwood/uniformerv2_demo/output_names2.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

pattern = r'A(\d{3})'

for line in lines:
    match = re.search(pattern, line)
    if match:
        a_number = int(match.group(1))
        output_names2list.append(a_number)


file_path = "/home/oakwood/uniformerv2_demo/output_2.txt"
test_output_labels = []

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    start_index = line.index(":") + 2
    end_index = line.index(",")
    extracted_text = line[start_index:end_index]
    test_output_labels.append(extracted_text.strip())

keys_from_dict = []

for value in output_names2list:
    if value in stgcn_labels_dict:
        keys_from_dict.append(stgcn_labels_dict[value])

# print(keys_from_dict)

t = 0
f = 0
print(len(keys_from_dict))
print(len(test_output_labels))
for i in range(len(keys_from_dict)-1):
    if keys_from_dict[i] == test_output_labels[i]:
        t +=1
    else:
        f +=1
print(t)
print(f)

file_path = "/home/oakwood/uniformerv2_demo/output_2.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

numerical_values = []

for line in lines:
    start_index = line.index(",") + 1
    numerical_value = float(line[start_index:].strip())
    numerical_values.append(numerical_value)

print(sum(numerical_values)/len(keys_from_dict))

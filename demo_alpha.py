uniformer = {}
stgcn = {}
stgcn_with_labels = {}

with open("/home/oakwood/uniformerv2_demo/readme.txt") as f:
    for line in f:
       (k, v) = line.split(',')
       uniformer[k] = float(v)

with open("/home/oakwood/uniformerv2_demo/stgcn_output.txt") as f:
    for line in f:
       (k, v) = line.split(',')
       stgcn[k] = float(v)

stgcn_labels_dict = {}
j = 0
with open("/home/oakwood/uniformerv2_demo/nturgbd_120.txt") as f:
    for line in f:
       stgcn_labels_dict[j] = line.rstrip('\n')
       j += 1

for i in stgcn:
    stgcn_with_labels[stgcn_labels_dict.get(int(i))] = stgcn[i]

#print(uniformer)
#print(stgcn_with_labels)
#print(stgcn_labels_dict)

def voting_algorithm(output1, output2):
    all_keys = set(output1.keys()) | set(output2.keys())

    votes = {}

    for key in all_keys:
        prediction1 = output1.get(key)
        prediction2 = output2.get(key)
        value1 = output1.get(key, 0)
        value2 = output2.get(key, 0)

        if prediction1 is not None:
            if (prediction1, key, "Uniformer") not in votes:
                votes[(prediction1, key, "Uniformer")] = 0
            votes[(prediction1, key, "Uniformer")] += value1

        if prediction2 is not None:
            if (prediction2, key, "STGCN") not in votes:
                votes[(prediction2, key, "STGCN")] = 0
            votes[(prediction2, key, "STGCN")] += value2

    final_prediction = max(votes, key=votes.get)
    prediction, key, output_name = final_prediction
    return prediction, key, output_name


with open('/home/oakwood/uniformerv2_demo/output_all.txt', 'a') as f:
    f.write(f"Voting: {voting_algorithm(uniformer, stgcn_with_labels)}, Uniformer: {max(uniformer, key=uniformer.get)} {uniformer[max(uniformer, key=uniformer.get)]}, STGCN: {max(stgcn_with_labels, key=stgcn_with_labels.get)} {stgcn_with_labels[max(stgcn_with_labels, key=stgcn_with_labels.get)]}\n")
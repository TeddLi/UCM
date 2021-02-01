import sys
import json
from collections import defaultdict

by_label_match = {"Reject": 0, "Accept": 0, "No Decision Yet": 0}
by_label_auto = {"Reject": 0, "Accept": 0, "No Decision Yet": 0}
by_label_gold = {"Reject": 0, "Accept": 0, "No Decision Yet": 0}

match = 0
exact_match = 0

total = 0
total_gold = 0
total_auto = 0

answers = {}
for line in open('dstc8-task3.gold.txt','r'):
    num = int(line.strip().split()[0])
    total += 1
    if 'No Decision Yet' in line:
        answers[num] = [(-1, "No Decision Yet")]
        total_gold += 1
        by_label_gold["No Decision Yet"] += 1
    else:
        for part in line.strip().split()[1:]:
            total_gold += 1
            msg, decision = part.split(":")
            by_label_gold[decision] += 1
            answers.setdefault(num, []).append((int(msg), decision))

output = {}
label_dict = {0 : "Neutral",1: "Reject", 2:"Accept" }
nums_dict = defaultdict(int)
start_index = 0
cur_index =0
cur_id = 0
for line in open('3', 'r'):
    id, sent_num, i_predicted_target, gold = line.split('\t')
    if int(id) not in output:
        output[int(id)] = []
        start_index = cur_index
    if int(i_predicted_target) !=0:
        output[int(id)].append((cur_index-start_index, label_dict[int(i_predicted_target)]))
        nums_dict[int(id)] += int(i_predicted_target)

    if cur_index - int(sent_num) +1 == start_index and nums_dict[int(id)] ==0:
        output[int(id)].append((-1, "No Decision Yet"))
    cur_index+=1






compare = output.copy()

# compare = json.load(sys.stdin)
for id, value in compare.items():
    num = id
    gold = answers[num]
    all_match = True
    to_print = [str(id)]
    for label in value:
        total_auto += 1
        pos = label[0]
        val = label[1]
        by_label_auto[val] += 1
        to_print.append(str(pos) +":"+ val)
        if (pos, val) in gold:
            match += 1
            by_label_match[val] += 1
        else:
            all_match = False
    if all_match and len(gold) == len(value):
        exact_match += 1
p = 100 * match / total_auto
r = 100 * match / total_gold
f = 2 * p * r / (p + r)
exact = 100 * exact_match / total

print("{:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(exact, p, r, f))


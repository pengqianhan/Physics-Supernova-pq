import re
edge = {
    'condition': 'x1[0] > 0'
}
condition = re.sub(r'(\w+)\[0\]', r'\1', edge['condition'])
print(condition)

edge1 = {
    'condition': 'x > 0'
}
condition1 = re.sub(r'(\w+)\[0\]', r'\1', edge1['condition'])
print(condition1)
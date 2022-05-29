N, M, K = map(int, input().split())

classes = []
for i in range(M):
    classes.append([])

Objects = list(map(int, input().split()))
for i in range(N):
    classes[Objects[i] - 1].append(i + 1)

groups = []
for i in range(K):
    groups.append([])

current = 0

for elements in classes:
    for element in elements:
        groups[current].append(element)
        current += 1
        current %= K

for group in groups:
    print(*[len(group), *group])

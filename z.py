for i in range(int(input())):
    n, x, y = list(map(int, input().split()))
    print(y//x, ' ', ' ')
    lst = list(map(int, input().split()))
    for xx in lst:
        x *= xx
        print(y//x, ' ', ' ')
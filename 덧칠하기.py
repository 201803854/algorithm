def solution(n, m, section):
    answer = 0
    count = 1
    start = section[0] + m -1
    for i in section :
        if i > start :
            count += 1
            start = i + m - 1
            
    return count
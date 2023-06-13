def solution(t, p):
    length = len(p) # 검정 문자열 길이
    pivot = 0 # 문자열 탐색 피벗
    answer = 0
    while pivot + length <= len(t) : # 검정할 문자열 길이 동안 검정
        temp = t[pivot:pivot + length] 
        if temp<= p :
            
            answer += 1
        pivot += 1

    return answer
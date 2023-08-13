def solution(k, m, score):
    answer = 0
    box_num = m
    score.sort(reverse=True)
    count = 0
    for i in score :
        
        temp = []
        temp.append(i)
        count+=1
        
        if count == m :
            answer += min(temp)*m
            count = 0
            
    return answer
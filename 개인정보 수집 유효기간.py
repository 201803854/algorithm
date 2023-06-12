def solution(today, terms, privacies):
    answer = []
    daytime = today.split('.')
    term_type = dict()
    privacy = []
    for i in terms:
        temp = i.split()
        term_type[temp[0]] = int(temp[1])  # term_type의 값들을 정수로 변환하여 저장합니다.
    
    for v, i in enumerate(privacies):
        i = i.replace(' ', '.')
        privacy = i.split('.')
        # 대여 기간을 계산하여 조건을 만족하는 경우에는 인덱스 v를 answer 리스트에 추가합니다.
        if (int(daytime[0]) - int(privacy[0])) * 12 + (int(daytime[1]) - int(privacy[1])) > term_type[privacy[3]]:
            
            answer.append(v+1)
        if (int(daytime[0]) - int(privacy[0])) * 12 + (int(daytime[1]) - int(privacy[1])) == term_type[privacy[3]]:
            if int(daytime[2])- int(privacy[2]) >= 0 :
                answer.append(v+1)
                

    
    return answer
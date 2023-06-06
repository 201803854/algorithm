def solution(cards1, cards2, goal):
    answer = 'Yes'
    hashmap1 = dict()
    hashmap2 = dict()
    temp1 = -1
    temp2 = -1
    for i,v in enumerate(cards1) : #순서 파악위해 딕셔너리 생성
        hashmap1[v] = i
    for i,v in enumerate(cards2) :
        hashmap2[v] = i
    for i in goal :
        
        if hashmap1.get(i,-1) != -1 : # 이전거보다 순서가 앞이면 no 저장
            if hashmap1[i] != temp1 +1 : # 처음 단어부터 시작하는지 확인
                answer = 'No'
            else:
                temp1 = hashmap1[i]
        else :
            if hashmap2[i] != temp2 +1 :
                answer = 'No'
            else:
                temp2 = hashmap2[i]
    return answer
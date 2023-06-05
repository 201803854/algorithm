def solution(keymap, targets):
    answer = [] #keymap split
    hashmap = dict() # 자판 누르는 횟수 저장
    result =[] #결과 저장
    for i in keymap : #keymap split
        answer.append(list(i))
    for i in answer: # 자판 문자당 누르는 횟수 설정
        for j,k in enumerate(i) :
            check = hashmap.get(k)
            if check == None :
                hashmap[k] = j+1
            else :
                if check > j+1 :
                    hashmap[k] = j+1
                else:
                    continue
    for i in targets : # 총 눌러야 하는 횟수 찾기
        
        ans = 0
        for j in i :
            temp = hashmap.get(j,-1)
            if temp == -1 :
                result.append(temp)
                break
            else :
                ans += temp
        if temp != -1 :
            result.append(ans)
        
    return result
def solution(name, yearning, photo):
    answer = []
    hashmap = dict()
    for i,v in enumerate(name):
        hashmap[v] = yearning[i] 
    
    for names in photo:
        yearn = 0
        for name in names:
            yearn += hashmap.get(name,0)
        
        answer.append(yearn)
        
    return answer        
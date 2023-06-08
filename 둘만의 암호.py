def solution(s, skip, index):
    answer = ''
    skip_set = set(skip)  # skip 문자열을 집합(set)으로 변환하여 효율적인 탐색을 위해 활용
    
    for char in s:
        count = 0
        check = 0
        replaced_char = char  # replaced_char 변수를 밖으로 이동
        
        while True:
            replaced_char = chr((ord(char) +1 + count - 97) % 26 + 97)

            if replaced_char not in skip_set:
                count += 1
            else:
                count += 1
                check += 1
            
            if count >= index + check:
                break
        replaced_char = chr((ord(char) + count - 97) % 26 + 97)
        answer += replaced_char
        
    return answer
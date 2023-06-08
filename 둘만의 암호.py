def solution(s, skip, index):
    answer = ''
    for i in s:
        count = sum(1 for c in skip if ord(i) < ord(c) <= ord(i) + index)
        
        replaced_char = chr((ord(i) + index + count - 97) % 26 + 97)
        answer += replaced_char
    return answer
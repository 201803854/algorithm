def solution(people, limit):
    # 사람들을 무게순으로 정렬
    people.sort()

    # 구명보트의 수를 저장할 변수 초기화
    answer = 0

    # 가장 가벼운 사람과 가장 무거운 사람을 가리키는 인덱스
    i, j = 0, len(people) - 1

    # 모든 사람이 보트를 탈 때까지 반복
    while i <= j:
        # 가장 가벼운 사람과 가장 무거운 사람이 함께 탈 수 있다면
        if people[i] + people[j] <= limit:
            # 가장 가벼운 사람도 보트에 태움
            i += 1

        # 가장 무거운 사람은 항상 보트에 태움
        j -= 1

        # 보트의 수를 1 증가
        answer += 1

    return answer  # 구명보트의 최소 개수 반환

def solution(park, routes):
    answer = []
    route = []
    
    for i in routes: # 루트 좌표와 이동거리로 변환
        route.append(i.split())

    for i, v in enumerate(park): # 시작좌표 찾기
        for k, j in enumerate(v):
            if j == 'S':
                start = [i, k]
                break

    current_pos = start
    
    for way in route: # 좌표, 이동거리 설정 후 길찾기
        direction, distance = way[0], int(way[1])
        next_pos = current_pos.copy()
        
        if direction == 'N': # 방향 설정
            next_pos[0] -= distance
        elif direction == 'S':
            next_pos[0] += distance
        elif direction == 'W':
            next_pos[1] -= distance
        elif direction == 'E':
            next_pos[1] += distance
        
        if next_pos[0] < 0 or next_pos[0] >= len(park) or next_pos[1] < 0 or next_pos[1] >= len(park[0]): #예외 설정
            continue
        
        valid_route = True
        
        if direction == 'N': # 장애물 찾기
            for i in range(current_pos[0] - 1, next_pos[0] - 1, -1):
                if park[i][current_pos[1]] == 'X':
                    valid_route = False
                    break
        elif direction == 'S':
            for i in range(current_pos[0] + 1, next_pos[0] + 1):
                if park[i][current_pos[1]] == 'X':
                    valid_route = False
                    break
        elif direction == 'W':
            for j in range(current_pos[1] - 1, next_pos[1] - 1, -1):
                if park[current_pos[0]][j] == 'X':
                    valid_route = False
                    break
        elif direction == 'E':
            for j in range(current_pos[1] + 1, next_pos[1] + 1):
                if park[current_pos[0]][j] == 'X':
                    valid_route = False
                    break

        if not valid_route:
            continue

        current_pos = next_pos  # 위치 업데이트
    
    
    return current_pos
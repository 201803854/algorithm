def solution(wallpaper):
    answer = []
    up = 0
    down = 0
    left = 0
    right = 0
    count = 0
    for i,j in enumerate(wallpaper):
        for v,k in enumerate(j) :
            if k == '#' :
                if count == 0 :
                    up = i
                    left = v
                    down = i+1
                    right = v+1
                    count += 1
                else :
                    if v < left :
                        left = v
                    if i+1 > down :
                        down = i +1
                    if v+1 > right :
                        right = v+1
    
    return [up,left,down,right]
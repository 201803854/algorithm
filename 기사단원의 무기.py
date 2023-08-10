def solution(number, limit, power):
    answer = 0
    power_list = []
    for knights in range(1,number+1) :
        
        count = 0
        
        for i in range(1,int(knights**(1/2))+1) :
                
            if i*i == knights:
                
                count += 1
            elif knights % i == 0 :
                count += 2
            
            else :
                continue
        
        if count > limit :
            power_list.append(power)
        else :
            power_list.append(count)
 
    return sum(power_list)

